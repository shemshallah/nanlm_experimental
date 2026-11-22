import os
for root, dirs, files in os.walk('/kaggle/input'):
    print('ROOT', root)
    print('DIRS', dirs[:10])
    print('FILES', files[:12])
