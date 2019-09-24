# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 23:17:55 2018

@author: megha
"""

import argparse
import cv2,time
import numpy as np
import sys
from scipy import misc


if __name__=="__main__":
	
	parser=argparse.ArgumentParser()
	parser.add_argument("--path",help="path")
	
	args=parser.parse_args()
	
	def input_function():
		global imge
		global ker
		if(len(sys.argv)< 2):
			image=cv2.VideoCapture(0)
			check,imge=image.read()
			cv2.imshow('Capturing',imge)		
			ker=cv2.waitKey(0)
			image.release()
		else:
			imge=cv2.imread("%s"%(args.path),1)				
			cv2.imshow('Image',imge)
			ker=cv2.waitKey(0) 
		
	input_function()	
	blue_img=np.copy(imge)
	green_img=np.copy(imge)
	red_img=np.copy(imge)
	blue_img[:,:,1:]=0
	green_img[:,:,(0,2)]=0
	red_img[:,:,:2]=0
	
	x,y,z=imge.shape
	
	
	
	def sliderHandler(n):
		global imge
		global dist
		kernel=np.ones((n,n),np.float32)/(n*n)
		dist=cv2.filter2D(imge,-1,kernel)
		cv2.imshow('processed',dist)
	
	def sliderHandler1(n):
		global imge
		global dist
		angle = 45*np.pi/180
		rows = imge.shape[0]
		cols = imge.shape[1]
		M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
		dist = cv2.warpAffine(imge,M,(cols,rows))
		cv2.imshow('Rotation',dist)
	
			
		
	def save(x):
		print("Image is saved")
		#if ker==ord('w'):
		cv2.imwrite("out.jpg",x)
		print("Press 'i' to reload image or press enter to quit!")
        

		
	def reload():
		cv2.destroyAllWindows()
		input_function()
		
		
		
	if ker==ord('g'):
		image_gray=cv2.cvtColor(imge,cv2.COLOR_BGR2GRAY)	
		cv2.imshow("Image_grayscale",image_gray)
		g=cv2.waitKey(0)
		if g==ord('w'):
			save(image_gray)
			m=cv2.waitKey(0)
			if(m==ord('i')):
				reload()
		if(g==ord('i')):
			reload()
			
	if(ker==ord('i')):
			reload()
			
	if(ker==ord('G')):	
		gray_image=np.copy(imge)
		gray_image[:] = imge.mean(axis=-1,keepdims=1) 
		cv2.imshow('User_implemented_Grayscale',gray_image)
		g=cv2.waitKey(0)
		if g==ord('w'):
			save(gray_image)
			m=cv2.waitKey(0)
			if(m==ord('i')):
				reload()
		if(g==ord('i')):
			reload()
		
		
	if(ker==ord('c')):		
		cv2.imshow('Blue Channel',blue_img)		
		r=cv2.waitKey(0)		
		if(r==ord('c')):		
			cv2.imshow('Green Channel',green_img)
			g=cv2.waitKey(0)
			if(g==ord('c')):		
				cv2.imshow('Red Channel',red_img)
				g=cv2.waitKey(0)
				cv2.destroyAllWindows()
				if g==ord('w'):
					save(blue_img)
					m=cv2.waitKey(0)
					if(m==ord('i')):
						reload()
				if(g==ord('i')):
					reload()
					
				
		
	if(ker==ord('s')):	
		imge=cv2.cvtColor(imge,cv2.COLOR_BGR2GRAY)
		cv2.imshow('blur',imge)
		cv2.createTrackbar('s','blur',0,10,sliderHandler)
		g=cv2.waitKey(0)
		if g==ord('w'):
			save(dist)
			m=cv2.waitKey(0)
			if(m==ord('i')):
				reload()
		if(g==ord('i')):
			reload()
		
	
		
	if (ker==ord('d')):
		print(imge.shape)
		lower_reso = cv2.resize(imge, (int(imge.shape[1]/2),int(imge.shape[0]/2)))
		print(lower_reso.shape)
		cv2.imshow('Modified_Image',lower_reso)
		g=cv2.waitKey(0)
		if g==ord('w'):
			save(lower_reso)
			m=cv2.waitKey(0)
			if(m==ord('i')):
				reload()
		if(g==ord('i')):
			reload()
	
	
	if (ker==ord('D')):
		print(imge.shape)
		lower_reso = cv2.pyrDown(imge)
		print(lower_reso.shape)
		cv2.imshow('Modified_Image',lower_reso)
		g=cv2.waitKey(0)
		if g==ord('w'):
			save(lower_reso)
			m=cv2.waitKey(0)
			if(m==ord('i')):
				reload()
		if(g==ord('i')):
			reload()
				
	if (ker==ord('x')):
		image_gray=cv2.cvtColor(imge,cv2.COLOR_BGR2GRAY)
		sobelx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype = np.float)
		sobely = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype = np.float)
		gx = cv2.filter2D(image_gray, -1, sobelx)
		gy = cv2.filter2D(image_gray, -1, sobely)
		g = np.sqrt(gx * gx + gy * gy)
		g *= 255.0 / np.max(g)
		cv2.imshow('x_derivative',gx)		
		g=cv2.waitKey(0)
		cv2.destroyAllWindows()
		if g==ord('w'):
			save(gx)
			m=cv2.waitKey(0)
			if(m==ord('i')):
				reload()
		if(g==ord('i')):
			reload()
		
		
	if (ker==ord('y')):
		image_gray=cv2.cvtColor(imge,cv2.COLOR_BGR2GRAY)
		sobelx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype = np.float)
		sobely = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype = np.float)
		gx = cv2.filter2D(image_gray, -1, sobelx)
		gy = cv2.filter2D(image_gray, -1, sobely)
		g = np.sqrt(gx * gx + gy * gy)
		g *= 255.0 / np.max(g)
		cv2.imshow('y_derivative',gy)		
		g=cv2.waitKey(0)
		cv2.destroyAllWindows()
		if g==ord('w'):
			save(gy)
			m=cv2.waitKey(0)
			if(m==ord('i')):
				reload()
		if(g==ord('i')):
			reload()
			
	if (ker==ord('m')):
		image_gray=cv2.cvtColor(imge,cv2.COLOR_BGR2GRAY)
		sobelx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype = np.float)
		sobely = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype = np.float)
		gx = cv2.filter2D(image_gray, -1, sobelx)
		gy = cv2.filter2D(image_gray, -1, sobely)
		
		g = np.sqrt(gx * gx + gy * gy)
		
		g *= 255.0 / np.max(g)
		print(g)
		
	if (ker==ord('r')):
		imge=cv2.cvtColor(imge,cv2.COLOR_BGR2GRAY)
		cv2.imshow('rotation',imge)
		cv2.createTrackbar('s','rotation',0,10,sliderHandler1)
		
		g=cv2.waitKey(0)
		if g==ord('w'):
			save(dist)
			m=cv2.waitKey(0)
			if(m==ord('i')):
				reload()
		if(g==ord('i')):
			reload()
		
	
	
	
	
	
	
	
	if (ker==ord('h')):
		print("--------------------------------------Description of the commands--------------------------------------------------------")
		print("----------------------------------Command Line Arguments---------------------------------------------------")
		print("We need to go to command prompt and give cmd as \"python\", followed by the relative filename and optional \"--path\" cmd argument followed by relative path name of the image. If path of image is not given then program will capture image from the web-cam.")
		print("--------------------------------------Shortcut-keys---------------------------------------------------------")
		print("'i'- reload the original image.")
		print("'w'- save the current image into the file 'out.jpg'")
		print("'g'- convert the image to grayscale using the openCV conversion function.")
		print("'G'- convert the image to grayscale using user defined implementation function.")
		print("'c'- cycle through the color channels of the image showing a different channel every time the key is pressed.")
		print("'s'- convert the image to grayscale and smoothing it using openCV function.")
		print("'S'- convert the image to grayscale and smoothing it using user defined function.")
		print("'d'- downsample the image by a factor of 2 without smoothing.")
		print("'D'- downsample the image by a factor of 2 with smoothing.")
		print("'x'- convert the image to grayscale and perform convolution with an x derivative filter.")
		print("'y'- convert the image to grayscale and perform convolution with an y derivative filter.")
		print("'m'- show the magnitude of the gradient normalized to the range [0,255].")
		print("'p'- convert the image to grayscale and plot the gradient vectors of the image every N pixels.")
		print("'r'- convert the image to grayscale and perform convolution with an y derivative filter.")
		print("'h'- displays short description of the program,its command line arguments, and the keys it supports.")
		
		
		
	
	
	
	