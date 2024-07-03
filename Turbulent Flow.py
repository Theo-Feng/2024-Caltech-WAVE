#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


a = 50e-6  
U0 = 178  
nu = 1e-6  
A = 0.239
k = 0.260


r_values = np.linspace(50e-6, 450e-6, 500)


first_term = (a**2) / (2 * r_values)
second_term = (k-A) * (7*nu/U0)**(1/5) * r_values**(4/5) / (80 * (A-2/9)**(4/5))
h_values = first_term + second_term


plt.figure(figsize=(10, 6))
plt.plot(r_values * 1e6, h_values * 1e6, label='h vs. r', color='green')
plt.xlabel('Radial Distance r (µm)')
plt.ylabel('Film Depth h (µm)')
plt.title('Variation of Film Depth h with Radial Distance r')
plt.legend()
plt.grid(True)
plt.savefig('Variation of Film Depth h with Radial Distance r')
plt.show()


# In[ ]:




