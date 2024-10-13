# Say "Hello, World!" With Python
if __name__ == '__main__':
    print("Hello, World!")

# Python If-Else
#!/bin/python3
import math
import os
import random
import re
import sys
if __name__ == '__main__':
    n = int(input().strip()) # Fixing the input line
    if n % 2 != 0:  # Checking if n is odd
        print("Weird")
    elif 6 <= n <= 20:  # Checking if n is between 6 and 20 inclusive
        print("Weird")
    else:  # All other even cases
        print("Not Weird")

# Arithmetic Operators
if __name__ == '__main__':
    a = int(input())
    b = int(input())
    
# Print the sum of the two numbers
print(a + b)
# Print the difference of the two numbers (first - second)
print(a - b)
# Print the product of the two numbers
print(a * b)

# Python: Division
if __name__ == '__main__':
    a = int(input())
    b = int(input())
    
# Print the result of integer division
print(a // b)
# Print the result of float division 
print(a / b)

# Loops
if __name__ == '__main__':
    n = int(input())
#Take all the non-negative integers that are less than n:
for i in range(n): 
#Print the square of each number
    print(i*i)

# Write a function
def is_leap(year):
    leap = False
    # If the year is divisible by 400, it is a leap year
    if year % 400 == 0:
        return True
    # If the year is divisible by 100 but not by 400, it is not a leap year
    elif year % 100 == 0:
        return False
    # If the year is divisible by 4 but not by 100, it is a leap year
    elif year % 4 == 0:
        return True
    # Otherwise, it is not a leap year
    else:
        return False

# Print Function
if __name__ == '__main__':
    n = int(input())
    for i in range (1, n+1):
      print(i,end='')

# List Comprehensions
if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())
    
    permutations = list([i,j,k] for i in range(x+1) for j in range(y+1) for k in range(z+1)  if i+j+k !=n)
    print( permutations)
        

# Find the Runner-Up Score!
if __name__ == '__main__':
    n = int(input())
    arr = list(map(int, input().split()))
    scores =[value for value in arr if ((-100 <= value <= 100) and (2 <= value <= 10))]
    arr_sorted= (sorted(set(arr)))
    print(arr_sorted[-2])    

# Nested Lists
if __name__ == '__main__':
  students = []
  for _ in range(int(input())):
        name = input()
        score = float(input())
        students.append([name, score])
#create a set for the scores in order to eliminate duplicates and then sort them.
  scores=sorted(set([x[1] for x in students]))
  second_lowest_score = scores[1]
  second_lowest_students = [x[0] for x in students if x[1]==second_lowest_score]
  second_lowest_students.sort()
  for students in second_lowest_students:
    print(students)

# Finding the percentage
if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
    query_name = input()
    sum= 0
    for i in range(len(student_marks[query_name])):
        sum= sum + student_marks[query_name][i]
    
    average = sum/len(student_marks[query_name])
    print(f"{average:.2f}")

# Lists
if __name__ == '__main__':
    N = int(input())
    List=[]
    for i in range(N):
        command=input().split();
        if command[0] == "insert":
            List.insert(int(command[1]),int(command[2]))
        elif command[0] == "append":
            List.append(int(command[1]))
        elif command[0] == "pop":
            List.pop();
        elif command[0] == "print":
            print(List)
        elif command[0] == "remove":
            List.remove(int(command[1]))
        elif command[0] == "sort":
            List.sort();
        else:
            List.reverse();

# Tuples
if __name__ == '__main__':
    n = int(input())
    integer_list = map(int, input().split())
    integer_tuple = tuple(integer_list) 
    result = hash(integer_tuple) 
    print(result) 

# sWAP cASE
def swap_case(s):
    string= s.swapcase()
    return string

# String Split and Join

def split_and_join(line):
    line=line.split(" ")
    line= "-".join(line)
    return line
if __name__ == '__main__':
    line = input()
    result = split_and_join(line)
    print(result)

# What's Your Name?
#
# Complete the 'print_full_name' function below.
#
# The function is expected to return a STRING.
# The function accepts following parameters:
#  1. STRING first
#  2. STRING last
#
def print_full_name(first, last):
    string= "Hello " + first + " " + last + "! You just delved into python."
    print(string)

# Mutations
def mutate_string(string, position, character):
    l= list(string)
    l[position]= character
    string= ''.join(l)
    return string

# Find a string
def count_substring(string, sub_string):
    tot=0
    for i in range(0, len(string)):
        if string[i:len(string)].startswith(sub_string):
            tot += 1
    return tot

# String Validators
if __name__ == '__main__':
    c = input()
    print ( any(s.isalnum() for s in c))
    print ( any(s.isalpha() for s in c))
    print ( any(s.isdigit() for s in c))
    print ( any(s.islower() for s in c))
    print ( any(s.isupper() for s in c))
    

# Text Alignment
def print_pattern(thickness, char='H'):
    # top
    top_part = [(char*i).rjust(thickness-1) + char + (char*i).ljust(thickness-1) for i in range(thickness)]
    print("\n".join(top_part))
    # center
    center_part = [(char*thickness).center(thickness*2) + (char*thickness).center(thickness*6) for _ in range(thickness+1)]
    print("\n".join(center_part))
    belt_part = [(char*thickness*5).center(thickness*6) for _ in range((thickness+1)//2)]
    print("\n".join(belt_part))
    print("\n".join(center_part))
    # bottom
    bottom_part = [((char*(thickness-i-1)).rjust(thickness) + char + (char*(thickness-i-1)).ljust(thickness)).rjust(thickness*6) for i in range(thickness)]
    print("\n".join(bottom_part))

if __name__ == '__main__':
    thickness = int(input())
    print_pattern(thickness)

# Text Wrap

def wrap(string, max_width):
    text= textwrap.fill(string, max_width)
    return text

# Designer Door Mat
n, m = map(int, input().split())

for i in range(n // 2):
    print('-' * ((m - (1 + 2 * i)*3) // 2) + '.|.' * (1 + 2 * i) + '-' * ((m - (1 + 2 * i)*3) // 2))
print('-' * ((m - len('WELCOME')) // 2) + 'WELCOME' + '-' * ((m - len('WELCOME')) // 2))

for i in range(n // 2 - 1, -1, -1):
    print('-' * ((m - (1 + 2 * i)*3) // 2) + '.|.' * (1 + 2 * i) + '-' * ((m - (1 + 2 * i)*3) // 2))

# String Formatting
def print_formatted(number):
    width = len(bin(number)[2:])
    for i in range(1, number+1):
        print(str(i).rjust(width), oct(i)[2:].rjust(width), hex(i)[2:].upper().rjust(width), bin(i)[2:].rjust(width))

# Alphabet Rangoli
def print_rangoli(size):
    # Define the alphabet
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    
    # Calculate the total width for formatting
    width = (4 * size) - 3  
    # Construct the top half of the rangoli
    for i in range(size):
        # Get the left part of the rangoli using the alphabet
        left_part = "-".join(alphabet[size - 1 - i:size])  
        
        # Create the complete line by mirroring the left part
        line = left_part[::-1] + left_part[1:]  
        
        # Print the line centered within the total width, filling with '-'
        print(line.center(width, '-'))  
    # Construct the bottom half of the rangoli
    for i in range(size - 2, -1, -1):
        # Get the left part of the rangoli using the alphabet
        left_part = "-".join(alphabet[size - 1 - i:size])  
        
        # Create the complete line by mirroring the left part
        line = left_part[::-1] + left_part[1:]  
        
        # Print the line centered within the total width, filling with '-'
        print(line.center(width, '-'))  
    

# Calendar Module
import calendar
month, day, year = list(map(int,input().split(' ')))
correct_day = calendar.weekday(year, month, day)
days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
print(days[correct_day].upper())

# Time Delta
#!/bin/python3
import math
import os
import random
import re
import sys
from datetime import datetime

def time_delta(t1, t2):
    t1 = datetime.strptime(t1, '%a %d %b %Y %H:%M:%S %z')
    t2 = datetime.strptime(t2, '%a %d %b %Y %H:%M:%S %z')
    delta= abs((t1-t2).total_seconds())
    return str(int(delta))
    
    
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    t = int(input())
    for t_itr in range(t):
        t1 = input()
        t2 = input()
        delta = time_delta(t1, t2)
        fptr.write(delta + '\n')
    fptr.close()

# Capitalize!


def solve(s):
    name = s.split(' ')
    for i in range(len(name)):
        name[i] = name[i].capitalize()
    return ' '.join(name)
    

# Introduction to Sets
def average(array):
    distint_heights = set(array)
    tot= len(distint_heights)
    sum_heights = sum(distint_heights)
    average = sum_heights/tot
    return average
        
    
    

# Symmetric Difference
M = int(input())
M_set = set(map(int,input().split(' ')))
N = int(input())
N_set = set(map(int,input().split(' ')))
union = M_set.union(N_set)
inter = M_set.intersection(N_set)
sym_dif = union.difference(inter)
numeri_ordinati = sorted(map(int, sym_dif))
for i in numeri_ordinati:
    print (i)
    


# Set .add()
N= int(input())
country = []
for i in range (N):
    country.append(input())

countries= set(country)
print (len(countries))

# Set .discard(), .remove() & .pop()
n = int(input())
s = set(map(int, input().split()))
N= int(input())
commands=[]
for i in range(N):
    commands.append(input().split(' '))
    
    if commands[i][0]== 'pop':
        s.pop()
    else:
        num= int(commands[i][1])
         
        if commands[i][0] == 'remove':
            s.remove(num)
        elif commands[i][0] == 'discard':
            s.discard(num)
print(sum(s))

# Set .union() Operation
n = int(input())
eng_stud = input().split(' ')
b = int(input())
fr_stud = input().split(' ')
eng_stud = set(eng_stud)
fr_stud = set(fr_stud)
stud = eng_stud.union(fr_stud)
print (len(stud))


# Set .intersection() Operation
n = int(input())
eng_roll_num = map(int, input().split(' '))
b = int(input())
fr_roll_num = map(int, input().split(' '))
eng_roll_num = set(eng_roll_num)
fr_roll_num = set(fr_roll_num)
enrolled = eng_roll_num.intersection(fr_roll_num)
print (len(enrolled))

# Set .difference() Operation
n = int(input())
eng_stud = map(int, input().split(' '))
m = int(input())
fr_stud = map(int, input().split(' '))  
eng_stud = set(eng_stud)
fr_stud = set(fr_stud)
only_fr = eng_stud.difference(fr_stud)
print(len(only_fr))

# Set .symmetric_difference() Operation
n = int(input())
eng_stud = map(int, input().split(' '))
m = int(input())
fr_stud = map(int, input().split(' '))  
eng_stud = set(eng_stud)
fr_stud = set(fr_stud)
only_fr = eng_stud.symmetric_difference(fr_stud)
print(len(only_fr))

# Set Mutations
n= int(input())
A= set( map(int, input().split(' ')))
N=int(input())
for i in range(N):
    op_and_lenght = input().split(' ')
    second_entry_as_int = int(op_and_lenght[1])
    elements= set(map(int, input().split(' ')))
    if op_and_lenght[0]== 'update':
        A.update(elements)
    
    elif op_and_lenght[0]== 'intersection_update':
        A.intersection_update(elements)
    elif op_and_lenght[0]== 'difference_update':
        A.difference_update(elements)
    elif op_and_lenght[0]== 'symmetric_difference_update':
        A.symmetric_difference_update(elements)
        
print(sum(A))
    
    
    


# Check Subset
T = int(input())
for i in range(T):
    n = int(input())
    A = set(input().split(' '))
    m = int(input())
    B = set(input().split(' '))
    print(A.intersection(B)==A) 
    

# Check Strict Superset
A = set(map(int, input().split()))
n = int(input())
for _ in range(n):
    X = set(map(int, input().split()))
    if A.issuperset(X) != True or len(A) == len(X): 
        print(False)
        break 
else: print(True)

# No Idea!
n, m = map(int, input().split())
arr = list(map(int, input().split()))
A = set(map(int, input().split()))
B = set(map(int, input().split()))
happiness=0
for i in range(len(arr)):
    if (arr[i] in A):
        happiness += 1
    elif (arr[i] in B):
        happiness -= 1
print(happiness)
    

# The Captain's Room
K = int(input())
rooms = list(map(int, input().split(' ')))
rooms_sum= sum(rooms)
rooms_set = set(rooms)
rooms_set_sum = sum(rooms_set)
rooms_families_sum = int(((rooms_sum - rooms_set_sum)/(K-1))*K)
room_cap = rooms_sum - rooms_families_sum
print(room_cap)


# collections.Counter()
from collections import Counter
X = int(input())
sizes_shop = list(map(int, input().split(' ')))
counter_shop = Counter(sizes_shop)
N = int(input())
list_desires = []
tot = 0
for i in range(N):
    size, price = map(int, input().split())
    if counter_shop[size]>0:
        counter_shop[size] -= 1
        tot += price
print(tot)
    
        
    
    
    
    

# DefaultDict Tutorial
from collections import defaultdict
n, m = map(int, input().split(' '))
d = defaultdict(list)
for i in range(n):
    d[input()].append(i+1)
    
for j in range(m):
    word = input()
    if word in d:
        print(' '.join(map(str, d[word])))
    else:
        print(-1)

# Collections.namedtuple()
from collections import namedtuple
N = int(input())
columns = input().split()
sum_marks = 0
for i in range(N):
    
    student = namedtuple('student', columns)
    MARKS, CLASS, NAME, ID = input().split()
    student_a = student(MARKS, CLASS, NAME, ID)
    sum_marks += int(student_a.MARKS)
    
print((sum_marks / N))
    
    

# Collections.OrderedDict()
from collections import OrderedDict
N= int(input())
items_dictionary = OrderedDict()
for i in range(N):
    items = input().rsplit(' ', 1)
    price = int(items[1])
    name = items[0]
    if name in items_dictionary:
        items_dictionary[name] += price
    else:
        items_dictionary[name] = price
for items, price in items_dictionary.items():
    print(items, price)

# Word Order
from collections import Counter
from collections import OrderedDict
n = int(input())
a = OrderedDict()
words=[]
for i in range(n):
    word= words.append(input())
words_counter = Counter(words)
counts = list(words_counter.values())
print(len(set(words)))
print(' '.join(map(str, counts)))
    


# Collections.deque()
from collections import deque
N = int(input())
d = deque()
for i in range (N):
    operations = list(input().split(' '))
    if operations[0]=='append':
        values = int(operations[1])
        d.append(values)
    elif operations[0]=='pop':
        d.pop()
    elif operations[0]=='popleft':
        d.popleft()
    elif operations[0]=='appendleft':
        values = int(operations[1])
        d.appendleft(values)
print(' '.join(map(str,d)))

# Piling Up!
from collections import deque
T = int(input())
for i in range(T):
    n = int(input())
    sideLengths = deque(map(int, input().split()))
    max_ = 2 ** 31
    valid_sequence = True
    for j in range(len(sideLengths)):
        if sideLengths[0] >= sideLengths[-1] and sideLengths[0] <= max_:
            max_ = sideLengths.popleft()
        elif sideLengths[-1] <= max_:
            max_ = sideLengths.pop()
        else:
            print("No")
            valid_sequence = False
            break
    if valid_sequence and not sideLengths:
        print("Yes")


# Exceptions
T= int(input())
for i in range(T):
    try:
        a, b = map(int, input().split(' '))
        print(a//b)
    except ZeroDivisionError as e:
        print("Error Code:", e) 
    except ValueError as e:
        print("Error Code:", e) 

# Zipped!
N, X = map(int, input().split())
 
marks_list = []
 
for i in range(X):
    marks = list(map(float, input().split()))
    marks_list.append(marks)
    marks_for_stud = zip(*marks_list)
for student_marks in marks_for_stud:
    avg = sum(student_marks) / X
    print(avg)

# Map and Lambda Function
cube = lambda x: x ** 3 
def fibonacci(n):
     fib_list = [] 
     for i in range(n):
          if i < 2:
            fib_list.append(i)
          else:
            fib_list.append(fib_list[-1] + fib_list[-2])
     return fib_list

# Standardize Mobile Number Using Decorators
def wrapper(f):
    def fun(l):
        f(["+91 " + num[-10:-5] + " " + num[-5:] for num in l])
    return fun

# Company Logo
#!/bin/python3
import sys
from collections import Counter
if __name__ == '__main__':
    s = input().strip()  
    letters_count = Counter(s)
    
    sorted_letters = sorted(letters_count.items(), key=lambda x: (-x[1], x[0]))[:3]
    for letter, count in sorted_letters:
        print(letter, count)

# Athlete Sort
#!/bin/python3
import math
import os
import random
import re
import sys

if __name__ == '__main__':
    nm = input().split()
    n = int(nm[0])
    m = int(nm[1])
    arr = []
    for _ in range(n):
        arr.append(list(map(int, input().rstrip().split())))
    k = int(input())
    
    sorted_arr = sorted(arr, key=lambda x: x[k])
    for athlete in sorted_arr:
        print(*athlete)

# ginortS
s = input()
lower = []
upper = []
odd_digits = []
even_digits = []
for char in s:
    if char.isnumeric():
        if int(char)%2 == 0:
            even_digits.append(char)
        else:
            odd_digits.append(char)
    else:
        if char.isupper():
            upper.append(char)
        else:
            lower.append(char)
lower.sort()
upper.sort()
odd_digits.sort()
even_digits.sort()
print(''.join(lower + upper + odd_digits + even_digits))

# Arrays

def arrays(arr):
    a = numpy.array(arr[::-1],float)
    return a

# Shape and Reshape
import numpy
line = list(map(int, input().split(' ')))
print (numpy.reshape(line,(3,3)))

# Transpose and Flatten
import numpy
N, M = map(int, input().split(' '))
rows = []
for i in range(N):
    rows.append(list(map(int, input().split(' '))))
array = numpy.array(rows)
print(numpy.transpose(array))
print (array.flatten())

# Concatenate
import numpy
N, M, P = map(int, input().split(' '))
rows_N = []
rows_M = []
for i in range(N):
    rows_N.append(list(map(int, input().split(' '))))
    
for i in range(M):
    rows_M.append(list(map(int, input().split(' '))))
array_N = numpy.array(rows_N)
array_M = numpy.array(rows_M)
print (numpy.concatenate((array_N, array_M), axis = 0))

# Zeros and Ones
import numpy
line = tuple(map(int, input().split(' ')))
print(numpy.zeros(line, dtype=int))
print(numpy.ones(line, dtype=int))

# Eye and Identity
import numpy
numpy.set_printoptions(legacy='1.13')
N, M = map(int, input().split(' '))
print (numpy.eye(N,M))

# Array Mathematics
import numpy
N, M = map(int, input().split(' '))
A = []
B = []
for i in range(N):
    A.append(list(map(int, input().split(' '))))
    
for i in range(N):
    B.append(list(map(int, input().split(' '))))
a = numpy.array(A, int)
b = numpy.array(B, int)
print (a + b)
print (a - b)     
print (a * b)
print(a//b)  
print(a % b)
print(a ** b)

# Floor, Ceil and Rint
import numpy
numpy.set_printoptions(legacy='1.13')
line = list(map(float, input().split(' ')))
my_array = numpy.array(line, float)
print(numpy.floor(my_array))
print(numpy.ceil(my_array))
print(numpy.rint(my_array))

# Sum and Prod
import numpy
N, M = map(int, input().split(' '))
A = []
for _ in range(N):
    A.append(list(map(int, input().split(' '))))
my_array = numpy.array(A)
sum_ = numpy.sum(my_array, axis=0)
product = numpy.prod(sum_)
print(product)

# Min and Max
import numpy
N, M = map(int, input().split(' '))
A = []
for _ in range(N):
    A.append(list(map(int, input().split(' '))))
my_array = numpy.array(A)
min_ = numpy.min(my_array, axis=1)
print(numpy.max(min_))

# Mean, Var, and Std
import numpy
import numpy
N, M = map(int, input().split(' '))
A = []
for _ in range(N):
    A.append(list(map(int, input().split(' '))))
    
my_array = numpy.array(A)
print(numpy.mean(my_array, axis = 1))
print(numpy.var(my_array, axis = 0))
print(round(numpy.std(my_array, axis=None), 11))

# Dot and Cross
import numpy
N = int(input())
A = []
B = []
for _ in range(N):
    A.append(list(map(int, input().split(' '))))
for _ in range(N):
    B.append(list(map(int, input().split(' '))))
my_array_A = numpy.array(A)
my_array_B = numpy.array(B)
print(numpy.dot(my_array_A, my_array_B))

# Inner and Outer
import numpy
A = numpy.array(list(map(int, input().split(' '))))
B = numpy.array(list(map(int, input().split(' '))))
print (numpy.inner(A, B))
print (numpy.outer(A, B))

# Polynomials
import numpy
coefficients = list(map(float, input().split(' ')))
x = float(input())
print(numpy.polyval(coefficients, x))

# Linear Algebra
import numpy

N = int(input())
A = []
for _ in range(N):
    A.append(list(map(float, input().split(' '))))
my_matrix = numpy.array(A)
det = numpy.linalg.det(my_matrix)
print(round(det,2))

# XML 1 - Find the Score

def get_attr_number(node):
    total_attributes = len(node.items())
    for child in node:
        total_attributes += get_attr_number(child)
    return total_attributes

# XML2 - Find the Maximum Depth

maxdepth = 0
def depth(elem, level):
    global maxdepth
    level += 1
    for i in elem:
        depth(i, level)
    if level > maxdepth:
        maxdepth = level

# Detect Floating Point Number
def check(s):
    try:
        float(s)
        return '.' in s and len(s.split('.')) == 2 and s[-1] != '.'
    except ValueError:
        return False
T = int(input())
for _ in range(T):
    N_ = input().strip()
    print(check(N_))

# Re.split()
regex_pattern = r"[,.]"	

# Group(), Groups() & Groupdict()
import re
S = input()
pattern = r'([a-zA-Z0-9])\1'
match = re.search(pattern, S)
if match:
    print(match.group(1))
else:
    print(-1)

# Re.findall() & Re.finditer()
import re
input_string = input()
escaped_string = re.escape(input_string)
vowel_set = 'aeiou'
consonant_set = 'qwrtypsdfghjklzxcvbnm'
pattern = r'(?<=[' + consonant_set + '])([' + vowel_set + ']{2,})(?=[' + consonant_set + '])'
found_matches = re.findall(pattern, escaped_string, flags=re.I)
if found_matches:
    print("\n".join(found_matches))
else:
    print(-1)

# Re.start() & Re.end()
import re
source_string = input()
search_pattern = input()
compiled_pattern = re.compile(search_pattern)
found_match = compiled_pattern.search(source_string)
if not found_match: 
    print('(-1, -1)')
while found_match:
    print('({0}, {1})'.format(found_match.start(), found_match.end() - 1))
    found_match = compiled_pattern.search(source_string, found_match.start() + 1)

# Regex Substitution
import re
def replace_operator(match):
    if match.group(1) == '&&':
        return 'and'
    return 'or'
number_of_cases = int(input())
for _ in range(number_of_cases):
    user_input = input()
    result = re.sub(r"(?<= )(\|\||&&)(?= )", replace_operator, user_input)
    print(result)

# Validating Roman Numerals
Thousand = 'M{0,3}'
Hundred = '(C[MD]|D?C{0,3})'
Ten = '(X[CL]|L?X{0,3})'
Digit = '(I[VX]|V?I{0,3})'
regex_pattern = r"%s%s%s%s$" % (Thousand, Hundred, Ten, Digit)

# Validating phone numbers
import re
N = int(input())
for i in range(N):
    if re.match(r'^[789]\d{9}$',input()):
        print("YES")
    else:
        print("NO")
        

# Validating and Parsing Email Addresses
import email.utils
import re
num_cases = int(input())
for _ in range(num_cases):
    input_email = input()
    extracted_email = email.utils.parseaddr(input_email)[1].strip()
    is_valid_email = bool(re.match(r"(^[A-Za-z][A-Za-z0-9._-]+)@([A-Za-z]+)\.([A-Za-z]{1,3})$", extracted_email))
    if is_valid_email:
        print(input_email)

# Hex Color Code
import re
num_lines = int(input())
for _ in range(num_lines):
    input_string = input()
    hex_colors = re.findall(r"(#[0-9A-Fa-f]{3}|#[0-9A-Fa-f]{6})(?:[;,.)]{1})", input_string)
    for color in hex_colors:
        if color:
            print(color)
            

# HTML Parser - Part 1
from html.parser import HTMLParser
class MyParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print("Start :", tag)
        for n, m in attrs:
            print("->", n, ">", m)
    def handle_startendtag(self, tag, attrs):
        print("Empty :", tag)
        for n, m in attrs:
            print("->", n, ">", m)
    def handle_endtag(self, tag):
        print("End   :", tag)
parser = MyParser()
for i in range(int(input())):
    parser.feed(input())

# HTML Parser - Part 2
from html.parser import HTMLParser
class MyHTMLParser(HTMLParser):
    def handle_comment(self, data):
        if '\n' not in data:
            print('>>> Single-line Comment')
            print(data)
        elif '\n' in data:
            print('>>> Multi-line Comment')
            print(data)
    def handle_data(self, data):
        if data != '\n':
            print('>>> Data')
            print(data)
html = ""       
for i in range(int(input())):
    html += input().rstrip()
    html += '\n'
    
parser = MyHTMLParser()
parser.feed(html)
parser.close()

# Detect HTML Tags, Attributes and Attribute Values
from html.parser import HTMLParser
class CustomHTMLParser(HTMLParser):
    def handle_starttag(self, tag_name, attributes):
        print(tag_name)
        for attribute in attributes:
            print('-> {} > {}'.format(attribute[0], attribute[1]))
    def handle_comment(self, data):
        pass
if __name__ == "__main__":
    parser_instance = CustomHTMLParser()
    number_of_lines = int(input())
    html_input = ''.join([input().strip() for _ in range(number_of_lines)])
    parser_instance.feed(html_input)

# Validating UID
import re
for _ in range(int(input())):
    u = ''.join(sorted(input()))
    try:
        assert re.search(r'[A-Z]{2}', u)
        assert re.search(r'\d\d\d', u)
        assert not re.search(r'[^a-zA-Z0-9]', u)
        assert not re.search(r'(.)\1', u)
        assert len(u) == 10
    except:
        print('Invalid')
    else:
        print('Valid')

# Validating Credit Card Numbers
import re
for _ in range(int(input())):
    user_input = input().strip()
    if re.match(r"^[456]\d{3}(-?\d{4}){3}$", user_input):
        clean_input = user_input.replace("-", "")
        if not re.search(r"(\d)\1{3}", clean_input):
            print("Valid")
        else:
            print("Invalid")
    else:
        print("Invalid")

# Validating Postal Codes
regex_integer_in_range = r"^[1-9]\d{5}$"
regex_alternating_repetitive_digit_pair = r"(\d)(?=\d\1)"

# Matrix Script
import re
n, m = map(int, input().split())
matrix = [input() for _ in range(n)]
traversed = [matrix[j][i] for i in range(m) for j in range(n)]
pattern = r'([a-z0-9])([^a-z0-9]+)([a-z0-9])'
decoded = re.sub(pattern, r'\1 \3', str.join('', traversed), 0, re.I) 
print(decoded)

# Decorators 2 - Name Directory

def person_lister(f):
    def inner(people):
        data = sorted(people, key=lambda person: int(person[2]))
        sorted_people = []
        for person in data:
            sorted_people.append(f(person))
        return sorted_people
    return inner

# The Minion Game
def minion_game(string):
    kevin_score = 0
    stuart_score = 0
    length = len(string)
    
    for i in range(length):
        if string[i] in 'AEIOU':
            kevin_score += length - i
        else:
            stuart_score += length - i
            
    if kevin_score > stuart_score:
        print(f"Kevin {kevin_score}")
    elif stuart_score > kevin_score:
        print(f"Stuart {stuart_score}")
    else:
        print("Draw")

    

# Merge the Tools!
def merge_the_tools(input_string, length):
    for index in range(0, len(input_string), length):
        substring = input_string[index:index + length]
        unique_chars = ''
        for char in substring:
            if char not in unique_chars:
                unique_chars += char
        print(unique_chars)


