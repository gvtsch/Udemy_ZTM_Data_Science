# %% [markdown]
# <a href="https://colab.research.google.com/github/gvtsch/Udemy_ZTM_Data_Science/blob/main/Python.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown]
# ## Datatypes

# %% [markdown]
# ### Fundamental Data Types
# 
# * `int`
# * `float`
# * `bool`
# * `str`
# * `list`
# * `tuple`
# * `set`
# * `dict`
# 

# %% [markdown]
# #### Integers

# %%
print(2+4)
print(type(2+4))
print(2*4)

# %% [markdown]
# #### Floats
# Need more memory to be stored

# %%
print(10.56)
print(type(10.56))
print(type(10.56+20))
print(type(9.9+1.1))

# %%
print(2 ** 3)
print(2 // 4) # returns Integer (rounded down)
print(5 // 4)

# %%
print(5%3)

# %% [markdown]
# #### Strings

# %%
print(type("Hello World!"))

# %%
username = "super_coder"
print(username)

# %%
long_string = """
WOW
O o
---
"""
print(long_string)

# %%
firstname = "Christoph"
lastname = "Kempkes"
print(firstname + "" + lastname)

# %% [markdown]
# #### String Concatenation

# %%
print("hello"+" "+"world!")

# %% [markdown]
# #### Type conversion

# %%
print(type(int(str(100))))

# %%
a=str(100)
b=int(a)
c=type(b)
print(c)

# %% [markdown]
# #### Escape sequence

# %%
weather = "It's sunny"
print(weather)
weather = 'It\'s sunny'
print(weather)
weather = "It\s kind of \"sunny\""
print(weather)

# %% [markdown]
# #### Formatted Strings

# %%
name = "Johnny"
age = 55
print("Hi " + name + ". You are "+ str(age) + " years old")
print("Hi {}. You are {} years old".format("Susann", "25"))
print("Hi {}. You are {} years old".format(name, age))
print("Hi {1}. You are {0} years old".format(age, name))
print("Hi {new_name}. You are {age} years old".format(age=21, new_name="Jack"))
print(f"Hi {name}. You are {age} years old")

# %% [markdown]
# #### String Indexes

# %%
selfish = "me you they"
print(selfish[0])
print(selfish[1])
print(selfish[2])
print(selfish[-1])
print(selfish[:5])
print(selfish[::1]) # This is called slicing
print(selfish[-5:])

# %% [markdown]
# #### Immutability
# Strings are immutable. I can not change values, once the string is created.

# %%
selfish = "01234567"
print(selfish)
selfish = 100
print(selfish)

# %% [markdown]
# #### Booleans

# %%
name = "Christoph"
is_cool = False
is_cool = True

# %% [markdown]
# ## Math Functions

# %%
print(round(3.1))
print(abs(-20))

# %% [markdown]
# ### Operator Precedence

# %%
print(20 + 3 * 4)
print((20 - 3) + 2 ** 2)

# %% [markdown]
# ## Variables
# 
# 

# %%
iq = 125
print(iq)

# %%
user_age=iq/7
a = user_age
print(a)

# %%
a, b, c = 1, 2, 3
print(b)

# %% [markdown]
# ### Constants

# %%
PI = 3.14 # Don't change this constant

# %% [markdown]
# ## Expressions vs Statements

# %%
iq = 100 # the hole line performing this action is the statement
user_age = iq/5 # iq/5 is an expression, it causes code; 

# %% [markdown]
# ## Augmented assignment operator

# %%
some_value = 5
some_value = some_value + 2
print(some_value)

# %%
some_value += 2
print(some_value)

# %% [markdown]
# ## Built in functions

# %%
print(len("hellloooooooo"))

# %%
greet="hellloooooooo"
print(greet[0:2])
print(greet[0:len(greet)])

# %% [markdown]
# ## Built in methods

# %%
quote = "to be or not to be. to be or not to be"
print(quote)
print(quote.capitalize())
print(quote.isnumeric())
print(quote.upper())
print(quote.replace("be", "me"))

# %% [markdown]
# ## Lists
# - List is an ordered sequence.
# - Lists are a form of arrays.
# - Lists are mutable

# %%
li = ["hello", "world"]
lis = [1, 2, 3, 4, 5]
list = [1, 2.5, "hello", False]
print(li)
print(lis)
print(list)

# %%
amazon_cart = ["notebook", "mouse", "sunglasses"]
print(amazon_cart[0])

# %% [markdown]
# ### List slicing

# %%
amazon_cart=[
    "notebook",
    "sunglasses",
    "toys",
    "grapes"
]
print(amazon_cart)
print(amazon_cart[0:2])
print(amazon_cart[1::2])

# %%
amazon_cart[0] = "Laptop"
print(amazon_cart)
print(amazon_cart[1:3])

# %%
new_cart = amazon_cart
new_cart[0] = "gum"
print(new_cart)
print(amazon_cart)
# Amazon cart aims somewhere in memory, new_cart aims there, too.

# %%
new_cart = amazon_cart[:]
# Create a copy using list slicing
new_cart[1] = "gummybear"
print(new_cart)
print(amazon_cart)

# %% [markdown]
# ### Matrix
# Multidimensional lists

# %%
matrix = [
    [1, 2, 3],
    [2, 4, 6],
    [7, 8, 9]
    ]
print(matrix)

# %%
print(matrix[1][1])

# %%
matrix = [
    [1, 0, 1],
    [0, 1, 0],
    [1, 0, 1]
]
print(matrix)

# %%


# %%



