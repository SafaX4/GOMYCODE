#!/usr/bin/env python
# coding: utf-8

# In[14]:


#question1
def max(a,b,c):
    if a>b and a>c: 
        return a
    elif b>a and b>c :
        return b
    else:
        return c
max(10,15,20)    
   
    
    



# In[25]:


#question2
def calculation (a,b):
    addition= a+b
    substraction = a-b
    return addition, substraction
res=calculation (50,30)
print(res)

    
    
    
    


# In[ ]:


#question3
def somme (l):
    add=0
    for i in l:
        add=add+i
    return add
def produit (l):
    prod=1
    for i in l:
        prod=prod*i
    return prod
liste = [1,5,4,6,8,2,12,13]
print(somme(liste))
print(produit(liste))
def extract (liste):
    liste_pair = []
    liste_impair = []

    for i in liste:
        if (i % 2 == 0):
            liste_pair.append (i) 
        
        else:
            liste_impair.append(i)
    return liste_pair , liste_impair
            
    
liste = [1,5,4,6,8,2,12,13]        
print (extract(liste))  
print ("liste pair" , extract(liste)[0])
somme (extract(liste)[0])
print(somme (extract(liste)[0]))
produit(extract(liste)[1])
print(produit (extract(liste)[1]))



# In[39]:






# In[ ]:




