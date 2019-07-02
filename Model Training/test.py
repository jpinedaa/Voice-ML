a = 5
b = 3
c = a/b
for i in range(20):
    temp_a = a
    temp_b = b
    a = 2*temp_a + temp_b
    b = temp_a + 2
    c = c + (a/b)

print(c)