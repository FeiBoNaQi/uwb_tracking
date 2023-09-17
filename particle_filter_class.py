# -*- coding: utf-8 -*-

import numpy as np
import copy
import matplotlib as plt
import sys
import math

class Particle(object):
    """
    Class of particle (cx, cy, sx, sy), corresponding to a rectangle bounding box
        cx: col index of the center of rectangle (pixel)
        cy: row index of the center of rectangle (pixel)


        The following attrs are optional
        weight: weight of this particle
        sigmas: transition sigmas of this particle
    """
    def __init__(self, cx=0, cy=0, sigmas=None):
        self.cx = cx
        self.cy = cy
        self.weight = 1
        if sigmas is not None:
            self.sigmas = sigmas
        else:
            self.sigmas = [0, 0]
        
    def transition(self, dst_sigmas=None, dst_interval=None):
        """
        transition by Gauss Distribution with 'sigma'
	Fill it !!!!!!!!!!!!!!!
        """
        if dst_sigmas is None:
            sigmas = self.sigmas
        else:
            sigmas = dst_sigmas
        
        if dst_interval is None:
            interval = 0.001
            sys.exit(1)
        else:
            interval = dst_interval
        
        dx = np.random.normal(0,sigmas[0])
        dy = np.random.normal(0,sigmas[1])
        self.cx = self.cx + dx*interval
        self.cy = self.cy + dy*interval
        return self
    
    def update_weight(self, w):
        self.weight = w

    def clone(self):
        """
        Clone this particle
        :return: Deep copy of this particle
        """
        return copy.deepcopy(self)
    
    def display(self):
        print('cx: {}, cy: {}'.format(self.cx, self.cy))
        print('weight: {} sigmas:{}'.format(self.weight, self.sigmas))

    def __eq__(self, ptc):
        return self.weight == ptc.weight

    def __le__(self, ptc):
        return self.weight <= ptc.weight


def transition_step(particles, sigmas, interval):
    """
    Sample particles from Gaussian Distribution
    :param particles: Particle list
    :param sigmas: std of transition model
    :return: Transitioned particles
    """
    #print('trans')
    for particle in particles:
        particle.transition(sigmas,interval)
    return particles


def weighting_step(particles, distance_diff_pre, tx_rx_pair,AnchorPos):
    """
    Compute each particle's weight by its distance to the predicted position
    :param particles: Current particles
    :param distance_diff_pre: predicted (tx_distance + rx_distance - tx_rx_distance)
    :param tx_rx_pair: which pair of transmitter and receiver
    :param AnchorPos: position of anchors
    :return: weights of particles
    """
    tx_pos = AnchorPos[tx_rx_pair[0]]
    rx_pos = AnchorPos[tx_rx_pair[1]]
    tx_rx_distance = math.sqrt(math.pow((tx_pos[0] - rx_pos[0]),2) + math.pow((tx_pos[1] - rx_pos[1]),2))
    weight = np.zeros(len(particles), dtype=np.float)
    i = 0
    sigma = 10
    scale = 1
    for particle in particles:
        # distance of particle to the anchor, and anchor to anchor
        tx_distance = math.sqrt(math.pow((particle.cx - tx_pos[0]),2) + math.pow((particle.cy - tx_pos[1]),2))
        rx_distance = math.sqrt(math.pow((particle.cx - rx_pos[0]),2) + math.pow((particle.cy - rx_pos[1]),2))
        distance_diff = tx_distance + rx_distance - tx_rx_distance

        # caulculate weight according to p(E|X) ;X is distance_diff;E is distance_diff_pre
        # normal distribution
        # weight[i] = 1/(sigma*math.sqrt(2*math.pi))*math.pow(math.e,-1/2*math.pow((distance_diff_pre-distance_diff),2)/(sigma*sigma))
        # Cauchy distribution
        weight[i] = 1/(math.pi*scale)*math.pow(scale,2)/(math.pow(scale,2)+math.pow(distance_diff_pre-distance_diff,2))
        i = i+1
    return weight


def resample_step(particles, weights, rsp_sigmas=None):
    """
    Resample particles according to their weights
    :param particles: Paricles needed resampling
    :param weights: Particles' weights
    :param rsp_sigmas: For transition of resampled particles
    """
    number_of_particles = len(particles)
    weights = weights / sum(weights) * number_of_particles
    c = np.zeros(number_of_particles)
    new_particles = []
    c[0] = weights[0]
    for i in range(1,number_of_particles):
        c[i] = c[i-1] + weights[i]
    u = np.zeros(number_of_particles+1) # extra 1 to make sure u[j+1] do not overflow
    u[0] = np.random.uniform(0,1)
    i = 0
    for j in range(0,number_of_particles):
        while(u[j] > c[i]):
            i = i+1
        new_particles.append(particles[i].clone())
        u[j+1] = u[j]+1
    return new_particles
    
def compute_similarities(features, template):
    """
    Compute similarities of a group of features with template
    :param features: features of particles
    :template: template for matching
    """
    pass
     
def compute_similarity(feature, template):
    """
    Compute similarity of a single feature with template
    :param feature: feature of a single particle
    :template: template for matching
    """
    pass

