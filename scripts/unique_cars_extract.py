import os
import cv2
import shutil

if __name__ == "__main__":

	unique_dir = "outputs/unique_tracks/"
	os.makedirs(unique_dir, exist_ok = True)


	for dir_elem in os.scandir("outputs/tracks/"):
		best_area = 0
		best_img_path = None
		if os.path.isdir(dir_elem.path): 
			for elem in os.scandir(dir_elem.path):
				img = cv2.imread(elem.path)
				area = img.shape[0] * img.shape[1]
				if(area > best_area):
					best_area = area
					best_img_path = elem.path

			target = os.path.join(unique_dir, dir_elem.name + ".png")
			shutil.copyfile(best_img_path, target)
			print(dir_elem.name, best_area)






