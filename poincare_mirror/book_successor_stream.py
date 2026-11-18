"""R62: persistent, strictly causal, book-isolated geometric successor stream.

Keeps only a bounded rolling prefix per book. A historical edge is indexed
only after its successor byte has been OBSERVED at the next input position.
Supports independently replayable snapshots and chunk-boundary continuity.
"""
from collections import defaultdict,deque
import math
import torch


class CausalBookSuccessor:
    VALID_MODES=("pair","pair_byte","pair_bigram","head_byte","hierarchical")

    def __init__(self,horizon=32768,vocab=257,key_mode="hierarchical",
                 value_control="exact",query_map=None,query_byte_map=None,
                 recency_half_life=0.0):
        if horizon<1 or vocab<2 or key_mode not in self.VALID_MODES:
            raise ValueError("Invalid book-memory horizon/vocab/key mode")
        if value_control not in ("exact","rotated"):
            raise ValueError("Invalid successor control")
        if (not isinstance(recency_half_life,(int,float)) or
                not math.isfinite(recency_half_life) or recency_half_life<0):
            raise ValueError('Invalid causal recency half-life')
        self.horizon,self.vocab,self.key_mode=horizon,vocab,key_mode
        self.recency_half_life=float(recency_half_life)
        self.value_control=value_control
        self.query_map=self._check_map(query_map,32640,"pair")
        self.query_byte_map=self._check_map(query_byte_map,vocab,"byte")
        self._vertices=torch.triu_indices(256,256,1).t().tolist()
        self.book_id=None
        self._clear()

    @staticmethod
    def _check_map(value,n,name):
        if value is None:
            return None
        values=value.tolist() if torch.is_tensor(value) else list(value)
        if len(values)!=n or any(not isinstance(v,int) or not 0<=v<n
                                 for v in values):
            raise ValueError("Invalid "+name+" query remapping")
        return values

    def _clear(self):
        self.position=0
        self.pending=None
        self.last_input=None
        self.exact=defaultdict(deque)
        self.head=defaultdict(deque)
        self.scheduled=deque()

    def begin_book(self,book_id):
        """Explicit reset prevents accidental reuse of another book's prefix."""
        if not isinstance(book_id,(str,int)) or str(book_id)=="":
            raise ValueError("Book identifier required")
        self._clear()
        self.book_id=str(book_id)

    def _key(self,address,byte,previous):
        if self.key_mode=="pair":
            return (address,)
        if self.key_mode=="pair_bigram":
            return (address,byte,previous) if previous is not None else None
        return (address,byte)

    def _install(self,at,addresses,byte,previous,successor):
        records=[]
        for edge,address in enumerate(addresses):
            rec={"u":at,"edge":edge,"pair":address,"byte":byte,
                 "previous":previous,"successor":successor}
            exact_key=self._key(address,byte,previous)
            if exact_key is not None:
                self.exact[exact_key].append(rec)
            head_keys=[]
            if self.key_mode in ("head_byte","hierarchical"):
                left,right=self._vertices[address]
                head_keys=((left,byte),(right,byte))
                for key in head_keys:
                    self.head[key].append(rec)
            records.append((exact_key,head_keys,rec))
        self.scheduled.append((at,records))

    def _expire(self,earliest):
        while self.scheduled and self.scheduled[0][0] < earliest:
            _,events=self.scheduled.popleft()
            for key,head_keys,record in events:
                if key is not None:
                    removed=self.exact[key].popleft()
                    if removed is not record:
                        raise AssertionError("Book index exact eviction order")
                    if not self.exact[key]:
                        del self.exact[key]
                for head_key in head_keys:
                    removed=self.head[head_key].popleft()
                    if removed is not record:
                        raise AssertionError("Book index head eviction order")
                    if not self.head[head_key]:
                        del self.head[head_key]

    def step(self,book_id,observed_byte,pair_ids):
        """Advance one observed input, then return distribution for its successor.

        The next *target* is NEVER an argument. At step t a pending entry
        u=t-1 becomes readable only once x[t] is supplied.
        """
        if self.book_id is None or str(book_id)!=self.book_id:
            raise ValueError("Begin the requested book before reading its history")
        if not isinstance(observed_byte,int) or not 0<=observed_byte<self.vocab:
            raise ValueError("Current observed byte outside vocabulary")
        pairs=pair_ids.tolist() if torch.is_tensor(pair_ids) else list(pair_ids)
        if not pairs or any(not isinstance(p,int) or not 0<=p<32640
                            for p in pairs):
            raise ValueError("Need valid canonical pair addresses")
        t=self.position
        if self.pending is not None:
            at,old_pairs,old_byte,old_previous=self.pending
            if at!=t-1:
                raise AssertionError("Nonconsecutive source continuation")
            self._install(at,old_pairs,old_byte,old_previous,observed_byte)
        earliest=max(0,t-self.horizon)
        self._expire(earliest)
        result=self._read_current(observed_byte,pairs,earliest)
        self.pending=(t,pairs[:],observed_byte,self.last_input)
        self.last_input=observed_byte
        self.position+=1
        return result

    def _read_current(self,observed_byte,pairs,earliest):
        """Shared strictly causal query for +1, +2 and +3 histories."""
        query_byte=(self.query_byte_map[observed_byte]
                    if self.query_byte_map is not None else observed_byte)
        query_prev=self.last_input
        matched=[]
        positions=set()
        for pair in pairs:
            q=(self.query_map[pair] if self.query_map is not None else pair)
            if self.key_mode=="pair_bigram" and query_prev is None:
                exact=[]
            else:
                exact=list(self.exact.get(self._key(q,query_byte,query_prev),()))
            if self.key_mode in ("pair","pair_byte","pair_bigram"):
                hits,confidence=exact,1.0
            else:
                a,b=self._vertices[q]
                shared={id(rec):rec for rec in self.head.get((a,query_byte),())}
                shared.update({id(rec):rec for rec in self.head.get((b,query_byte),())})
                hits,confidence=(exact,1.0) if (
                    self.key_mode=="hierarchical" and exact) else (
                    list(shared.values()),.25)
            if hits:
                matched.append((hits,confidence))
                positions.update(rec["u"] for rec in hits)
        # Thousands of geometric hits can accumulate on one byte. Compute
        # unit mass in float64 before converting for the trained joint rail.
        hist=torch.zeros(self.vocab,dtype=torch.float64)
        coverage=0.0
        if matched:
            coverage=sum(confidence for _,confidence in matched)/len(pairs)
            for hits,_ in matched:
                if self.recency_half_life:
                    # Stable normalization: only observed, already installed
                    # episodes contribute, and the newest hit has weight 1.
                    newest=max(rec['u'] for rec in hits)
                    weights=[2.0**(-(newest-rec['u'])/self.recency_half_life)
                             for rec in hits]
                    total_weight=sum(weights)
                else:
                    weights=[1.0]*len(hits)
                    total_weight=float(len(hits))
                for rec,w in zip(hits,weights):
                    # R54 wrong-value oracle rotates the observed prefix:
                    # earliest u receives x[t], later u receives x[u].
                    value=(observed_byte if rec["u"]==earliest else rec["byte"]) if (
                        self.value_control=="rotated") else rec["successor"]
                    hist[value]+=w/(len(matched)*total_weight)
            hist/=hist.sum()
        hist=hist.to(torch.float32)
        return hist,coverage,float(len(positions))

    def read_chunk(self,book_id,observed,pairs):
        """Batched over time, not across books; preserves persistent prefix."""
        if observed.ndim!=1 or pairs.ndim!=2 or pairs.shape[0]!=observed.shape[0]:
            raise ValueError("Need one book's T inputs and T,E geometric pairs")
        if observed.device.type!="cpu" or pairs.device.type!="cpu":
            raise ValueError("Persistent book index is CPU-only")
        rows=[self.step(book_id,int(byte),address.tolist())
              for byte,address in zip(observed.tolist(),pairs)]
        return (torch.stack([r[0] for r in rows]),
                torch.tensor([r[1] for r in rows],dtype=torch.float32),
                torch.tensor([r[2] for r in rows],dtype=torch.float32))

    def state_dict(self):
        """Portable replay state, without optimizer or neural-model mutation."""
        if self.book_id is None:
            raise ValueError("No book currently active")
        records=[[(rec["pair"],rec["byte"],rec["previous"],
                   rec["successor"]) for _,_,rec in events]
                 for _,events in self.scheduled]
        return {"book_id":self.book_id,"position":self.position,
                "horizon":self.horizon,"vocab":self.vocab,
                "key_mode":self.key_mode,"value_control":self.value_control,
                "recency_half_life":self.recency_half_life,
                "pending":self.pending,"last_input":self.last_input,
                "first_position":(self.scheduled[0][0]
                                  if self.scheduled else self.position),
                "completed":records}

    def load_state_dict(self,state):
        for field in ("horizon","vocab","key_mode","value_control"):
            if state[field]!=getattr(self,field):
                raise ValueError("Incompatible stream snapshot "+field)
        if state.get('recency_half_life',0.0)!=self.recency_half_life:
            raise ValueError('Incompatible stream snapshot recency_half_life')
        if state["position"]<0:
            raise ValueError("Invalid stream position")
        self.begin_book(state["book_id"])
        for offset,group in enumerate(state["completed"]):
            at=state["first_position"]+offset
            if at>=state["position"]:
                raise ValueError("Completed future history in snapshot")
            if not group:
                raise ValueError("Empty historical pair group")
            pairs=[record[0] for record in group]
            observed=group[0][1]
            previous=group[0][2]
            successor=group[0][3]
            if any((r[1],r[2],r[3])!=(observed,previous,successor)
                   for r in group):
                raise ValueError("Inconsistent snapshot historical group")
            self._install(at,pairs,observed,previous,successor)
        self.position=state["position"]
        self.pending=state["pending"]
        self.last_input=state["last_input"]


class CausalLagBookSuccessor(CausalBookSuccessor):
    """Causal historical +2/+3 successors under the SAME geometric keys.

    A source at s becomes readable only when the true x[s+lag] has appeared
    in the past or current input, never from a future label.
    """
    def __init__(self, *, lag, **kwargs):
        if lag not in (2,3):
            raise ValueError('Lagged book recall supports +2 and +3 only')
        super().__init__(**kwargs)
        self.lag=lag
        self.awaiting=deque()

    def begin_book(self,book_id):
        super().begin_book(book_id)
        self.awaiting=deque()

    def step(self,book_id,observed_byte,pair_ids):
        if self.book_id is None or str(book_id)!=self.book_id:
            raise ValueError('Begin requested book before lagged retrieval')
        if not isinstance(observed_byte,int) or not 0<=observed_byte<self.vocab:
            raise ValueError('Current observed byte outside vocabulary')
        pairs=pair_ids.tolist() if torch.is_tensor(pair_ids) else list(pair_ids)
        if not pairs or any(not isinstance(p,int) or not 0<=p<32640 for p in pairs):
            raise ValueError('Need valid canonical pair addresses')
        t=self.position
        if len(self.awaiting)==self.lag:
            at,old_pairs,old_byte,old_prev=self.awaiting.popleft()
            if at!=t-self.lag:
                raise AssertionError('Lagged episode matured at wrong timestep')
            self._install(at,old_pairs,old_byte,old_prev,observed_byte)
        earliest=max(0,t-self.horizon)
        self._expire(earliest)
        result=self._read_current(observed_byte,pairs,earliest)
        self.awaiting.append((t,pairs[:],observed_byte,self.last_input))
        self.last_input=observed_byte
        self.position+=1
        return result

    def state_dict(self):
        state=super().state_dict()
        state.update(lag=self.lag,awaiting=list(self.awaiting))
        return state

    def load_state_dict(self,state):
        if state.get('lag')!=self.lag:
            raise ValueError('Lagged snapshot horizon mismatch')
        super().load_state_dict(state)
        self.awaiting=deque(state['awaiting'])
        if len(self.awaiting)>self.lag:
            raise ValueError('Too many pending lagged episodes')
