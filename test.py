import numpy as np

a = np.array([[1,1,1], 
              [2,2,2], 
              [3,3,3]])

b = np.array([[50,50,50], 
              [60,60,60], 
                        ])

c = np.array([[50,50   ], 
              [60,60   ], 
              [70,70   ]])

d = np.array([[50,50   ], 
              [60,60   ], 
                        ])

print(np.int8 == np.uint8)

from itertools import permutations, product, combinations, combinations_with_replacement
for x in product("ACGT", "ACGT", "ACGT", "ACGT"):
    print(x)

for x in combinations("ACTG", 2):
    print(x)

# for x in combinations_with_replacement("ACTG", 2):
#     print(x)

print(np.diagonal(a, -1).shape[0])

a = np.array([3], dtype=np.float32)
b = np.array([2], dtype=np.uint8)
a[0] = b[0] * 999
print(a)