#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().run_line_magic('pylab', 'inline')


# In[2]:


from tinc import *
import os
import numpy as np


# In[3]:



A = Parameter("A", "weighting")
A.values = np.linspace(.1,2,3)

#A.ids = ["dir1", "dir2", "dir3", "dir4", "dir5"]

B = Parameter("B", "weighting")
B.values = np.linspace(0,2,3)

C = Parameter("C", "weighting")
C.values = np.linspace(.01,.1,3)

graph_buffer = DiskBufferImage("graph", "out.png", "graph_output")



# In[4]:


ps = ParameterSpace("ps")
ps.register_parameters([A, B, C])
ps.enable_cache()


# In[5]:


def make_graph(A, B, C):
    #print("Parameter value " + str(parameter_value))
    #data = [random.random() * C for i in range(int(A))]
    
    #hard-coding this now, change later
    ce_fits_dir = '/Volumes/DeoResearch/experiments/rocksalt3_ZrN_casm/ce_fits/'
    prefix = 'rocksalt3_ZrN_casm_'
    
    

    #fname = "out.png"
    fit_dir = 'A_%s_B_%s_kt_%s' % ((A), (B), (C))
    image_name = prefix + fit_dir + '.png'
    figure(figsize=10)
    #title(f" B = {B}")
    #plot(data)
    #savefig(fname)
    close() # Avoid showing as an additional graph in jupyter
    image_path = os.path.join(ce_fits_dir, fit_dir, image_name)
    file = open(image_path, 'rb')
    return file.read()

def value_changed(value):
    imagedata = ps.run_process(make_graph)
    graph_buffer.data = imagedata

A.register_callback(value_changed)
B.register_callback(value_changed)
C.register_callback(value_changed)


# In[6]:


from ipywidgets import GridspecLayout
grid = GridspecLayout(3, 2, height='300px')
grid[:2, 1] = graph_buffer.interactive_widget()
grid[0, 0] = A.interactive_widget()
grid[1, 0] = B.interactive_widget()
grid[2, 0] = C.interactive_widget()
grid


# In[ ]:





# In[ ]:





# In[7]:


ps.sweep(make_graph)


# In[8]:


#ps.set_current_path_template("%%A:ID%%/c_%%C%%")


# In[9]:


#ps.get_current_relative_path()


# In[ ]:





# In[10]:


#print(graph_buffer.data)


# In[ ]:




