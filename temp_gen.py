"""
Generate cooling schedules for annealing
"""
import numpy

def constant_sched (T0, length):
  A = numpy.linspace(T0,1,length)
  return A

def constant_sched_trailing(T0, length):
  A = numpy.linspace(T0,1,length)
  A = numpy.append(A, [1.]*20)
  #numpy.append(A, [1]*(length/10))
  return A
