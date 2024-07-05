#Creating a class for data augmentation
import  cv2
import imgaug
import imageio
import os
import numpy as np
from imgaug import augmenters as iaa
import glob

class DataAugmentation:
	def __init__(self, root_dir="",output_dir=""):
		self.root_dir = root_dir
		self.output_dir = output_dir
		print("Instance of the DataAugmentation class created")
	def augmentation_of_image(self, test_image, output_path):
		self.test_image = test_image
		self.output_path = output_path
		#define the Augmenters
		#properties: A range of values signifies that one of these numbers is randmoly chosen for every augmentation for every batch

		# Apply affine transformations to each image.
		rotate = iaa.Affine(rotate=(-90,90));  
		scale = iaa.Affine(scale={"x": (0.5, 0.9), "y": (0.5,0.9)}); 
		translation = iaa.Affine(translate_percent={"x": (-0.15, 0.15), "y": (-0.15, 0.15)})
		shear = iaa.Affine(shear=(-2, 2)); 
		zoom = iaa.PerspectiveTransform(scale=(0.01, 0.15), keep_size=True) 
		h_flip = iaa.Fliplr(1.0); 
		v_flip = iaa.Flipud(1.0); 
		padding=iaa.KeepSizeByResize(iaa.CropAndPad(percent=(0.05, 0.25)))


		#More augmentations
		blur = iaa.GaussianBlur(sigma=(0, 1.22)) 
		contrast = iaa.contrast.LinearContrast((0.75, 1.5))
		contrast_channels = iaa.LinearContrast((0.75, 1.5), per_channel=True) 
		sharpen = iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5))
		gauss_noise = iaa.AdditiveGaussianNoise(scale=0.111*255, per_channel=True) 
		laplace_noise = iaa.AdditiveLaplaceNoise(scale=(0, 0.111*255)) 


		#Brightness 
		brightness = iaa.Multiply((0.35,1.65)) 
		brightness_channels = iaa.Multiply((0.5, 1.5), per_channel=0.75) 

		#CHANNELS (RGB)=(Red,Green,Blue)
		red =iaa.WithChannels(0, iaa.Add((10, 100))) 
		red_rot = iaa.WithChannels(0,iaa.Affine(rotate=(0, 45))) 
		green= iaa.WithChannels(1, iaa.Add((10, 100)))
		green_rot=iaa.WithChannels(1,iaa.Affine(rotate=(0, 45))) 
		blue=iaa.WithChannels(2, iaa.Add((10, 100)))
		blue_rot=iaa.WithChannels(2,iaa.Affine(rotate=(0, 45))) 

		#colors
		channel_shuffle =iaa.ChannelShuffle(1.0); #shuffle all images of the batch
		grayscale = iaa.Grayscale(1.0)
		hue_n_saturation = iaa.MultiplyHueAndSaturation((0.5, 1.5), per_channel=True) #change hue and saturation with this range of values for different values 
		add_hue_saturation = iaa.AddToHueAndSaturation((-50, 50), per_channel=True) #add more hue and saturation to its pixels
		#Quantize colors using k-Means clustering
		kmeans_color = iaa.KMeansColorQuantization(n_colors=(4, 16)) #quantizes to k means 4 to 16 colors (randomly chosen). Quantizes colors up to 16 colors

		#Alpha Blending 
		blend =iaa.AlphaElementwise((0, 1.0), iaa.Grayscale((0,1.0)))  #blend depending on which value is greater

		#Contrast augmentors
		clahe = iaa.CLAHE(tile_grid_size_px=((3, 21),[0,2,3,4,5,6,7])) #create a clahe contrast augmentor H=(3,21) and W=(0,7)
		histogram = iaa.HistogramEqualization() #performs histogram equalization

		#Augmentation list of metadata augmentors
		OneofRed = iaa.OneOf( [red])
		OneofGreen = iaa.OneOf( [green] )
		OneofBlue = iaa.OneOf( [blue])
		contrast_n_shit = iaa.OneOf([contrast, brightness, brightness_channels])
		SomeAug = iaa.SomeOf(2,[rotate,scale, translation, shear, h_flip,v_flip],random_order=True)
		SomeClahe = iaa.SomeOf(2, [clahe, iaa.CLAHE(clip_limit=(1, 10)),iaa.CLAHE(tile_grid_size_px=(3, 21)),iaa.GammaContrast((0.5, 2.0)),
                            iaa.AllChannelsCLAHE() , iaa.AllChannelsCLAHE(clip_limit=(1, 10), per_channel=True)],random_order=True) #Random selection from clahe augmentors
		edgedetection= iaa.OneOf([iaa.EdgeDetect(alpha=(0, 0.7)),iaa.DirectedEdgeDetect(alpha=(0, 0.7), direction=(0.0, 1.0))])# Search in some images either for all edges or for directed edges.These edges are then marked in a black and white image and overlayed with the original image using an alpha of 0 to 0.7.
		canny_filter = iaa.OneOf([iaa.Canny(), iaa.Canny(alpha=(0.5, 1.0), sobel_kernel_size=[3, 7])]) #choose one of the 2 canny filter options
		OneofNoise = iaa.OneOf([blur, gauss_noise, laplace_noise])
		Color_1 = iaa.OneOf([channel_shuffle,grayscale, hue_n_saturation , add_hue_saturation, kmeans_color])
		Color_2 = iaa.OneOf([channel_shuffle,grayscale, hue_n_saturation , add_hue_saturation, kmeans_color])
		Flip = iaa.OneOf([histogram , v_flip, h_flip])

		#Define the augmentors used in the Data Augmentation
		Augmentors= [SomeAug, SomeClahe, SomeClahe, edgedetection,sharpen, canny_filter, OneofRed, OneofGreen, OneofBlue, OneofNoise, Color_1, Color_2, Flip, contrast_n_shit]


		for i in range(0,14):
			img = cv2.imread(test_image) 
			images = np.array([img for _ in range(14)], dtype=np.uint8)
			#print(images.shape)  # Prints: (14, 224, 224, 3)
			images_aug = Augmentors[i](images=images)  
			cv2.imwrite(os.path.join(output_path,test_image +"new"+str(i)+'.jpg'), images_aug[i]) 
class DataAugmentation_Extension:
	def __init__(self, directory=""):
		self.directory = directory
		print("Instance of DataAugmentation_Extension class created")

	def printnow(self, dir):
		print("Just testing that the method calling is working "+ dir)


	def extend_dataset(self,directory):
		#Create an instance of class 
		print("HEY")
		library_augment= DataAugmentation()
		self.directory = directory 
		if not os.path.exists(self.directory):
			print("ERROR! Couldn't find directory!")
		else:
			print("Directory exists")
		for file in os.listdir(directory):            #for any file inside the root directory 
			classes_path = os.path.join(directory, file)  #So for every folder class we create a class directory
			class_files = [name for name in glob.glob(os.path.join(classes_path,'*.BMP'))]  #alternatively we can use the globe as mentioned
			#print(class_files) #call augmentation for all class_files
			for i in range(len(class_files)):
				library_augment.augmentation_of_image(class_files[i], classes_path)