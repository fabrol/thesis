"""
Generate cooling schedules for annealing
"""
import numpy

def constant_sched (T0, length):
  A = numpy.linspace(T0,1,length)
  return A
