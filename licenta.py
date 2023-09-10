# importa pachetele necesare
import imutils
import time
import cv2
import matplotlib.pyplot as plt

# --------importa pentru modelul de incarcare-------->q
from skimage import feature
from skimage.feature import local_binary_pattern
import numpy as np
from sklearn.externals import joblib
# ------------------------------------->

class LocalBinaryPatterns:
	def __init__(self, numPoints, radius):
		# memoreaza numarul de puncte si raza
		self.numPoints = numPoints
		self.radius = radius
 
	def describe(self, image, eps=1e-7):
		# calculeaza reprezentarea modelului binar local din imagine și apoi utilizeaza reprezentarea LBP,
		# pentru a construi histograma modelelor
		lbp = feature.local_binary_pattern(image, self.numPoints,
			self.radius, method="uniform")
		(hist, _) = np.histogram(lbp.ravel(),
			bins=np.arange(0, self.numPoints + 3),
			range=(0, self.numPoints + 2))
 
		# normalizeaza histograma
		hist = hist.astype("float")
		hist /= (hist.sum() + eps)
 
		# returneaza histograma tiparelor binare locale
		return hist

 
# inițializeaza primul cadru din fluxul video
firstFrame = None
min_area = 100 #defaul 100
bgThresh = 25 #defaul 25
# incarca modelul din fișier
joblib_file = "joblib_model.pkl" 
model = joblib.load(joblib_file)
min_variance = 100

cap = cv2.VideoCapture('carlibaba.mp4')

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Verifica daca camera s-a deschis cu succes
if (cap.isOpened()== False): 
	print("Error opening video stream or file")
# bucla peste cadrele videoclipului
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('delta.mp4', -1 , 20.0, (500,int(frame_height*500/frame_width)))

radius = 3
numPoints = 8*radius 
desc = LocalBinaryPatterns(numPoints, radius)
num = 1
while (cap.isOpened()):
	# ia cadrul curent și inițializați cel ocupat/neocupat
# text
	start_time = time.time()
	ret, frame = cap.read()
 
	# dacă cadrul nu a putut fi apucat, atunci am ajuns la capăt de videoclip
	if frame is None:
		break

 
    # Afișați cadrului original
	cv2.imshow('Frame',frame)

	# redimensioneaza cadrul, il converteste in tonuri de gri si il estompeaza
	frame = imutils.resize(frame, width=500)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

 
	# if the first frame is None, initialize it
	if firstFrame is None:
		firstFrame = gray
		continue
	
	# calculeaza diferența absolută dintre cadrul curent și primul cadru
	frameDelta = cv2.absdiff(firstFrame, gray)
	thresh = cv2.adaptiveThreshold(frameDelta, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 1)

	# Calcularea diferenței absolute între cadrele consecutive
	variance = np.var(frameDelta)
	# dilateaza imaginea cu prag pentru a umple găurile, apoi gaseste contururi pe imaginea cu prag
# --------------------------CHECK BY COLOR SPACE--------------------------->
	lower_threshold = np.array([0, 0, 101])  # Rosu > 100, verde < 50, albastru < 50
	upper_threshold = np.array([49, 49, 255])  # rosu pana la 255, verde si albastru pana la 49

	# foloseste cv2.inRange() pentru a creea o masca binara unde 255 (alb) reprezentand pixelilor în cadrul pragului
	mask = cv2.inRange(frame, lower_threshold, upper_threshold)

	# aplica masca pe gama de culoare dorita
	threshCheckColor = cv2.bitwise_and(thresh, thresh, mask=mask)	

# ------------------------------------------------------------------------->
	histograme = []
	thresh = cv2.dilate(threshCheckColor, None, iterations=5)
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	points = []
	# Se verifica daca dispersia trece de pragul impus
	if variance > min_variance:
		# buclă peste contururi
		for c in cnts:
			# daca conturul e prea mic, se ignora
			if cv2.contourArea(c) < min_area:
				continue

			(x, y, w, h) = cv2.boundingRect(c)
	# ------------------------------PREDICTIE------------------------------->
			crop_img = frame[y:y+h, x:x+w]	
			dst = cv2.resize(crop_img,(110,110))
			# dst = cv2. cvtColor(dst, cv2.COLOR_BGR2HSV)
			b, g, r = cv2.split(dst)

			histB = desc.describe(b)
			histG = desc.describe(g)
			histR = desc.describe(r)

			
			hist = np.concatenate((histB,histG,histR))
			prediction_ = model.predict(hist.reshape(1, -1))
			if(prediction_[0] != 0):
				# hb=[]
				# hg=[]
				# hr=[]
				# for i, col in enumerate(['b', 'g', 'r']):
				# 	hists = cv2.calcHist([dst], [i], None, [256], [0, 256])
				# 	plt.plot(hists, color=col)
				# 	plt.xlim([0, 256])
				# plt.show(block=False)
				# plt.pause(1)
				# plt.close()
				# histograme.append([hb,hg,hr])
				for contour in c:
					# Iterați peste punctele de contur
					for point in contour:
						x, y = point.ravel()  # Despachetați valorile punctelor de contur
						points.append((x, y))
						
			else:
				firstFrame = gray
	# ------------------------------PREDICTIE------------------------------->
		
	# Convert the points list to a NumPy array of type np.int32
	# if len(histograme)!= 0:
	# 	# plt.figure(figsize=(12, 4))
	# 	histbb = []
	# 	histgg = []
	# 	histrr = []
	# 	for b, g, r in histograme:
	# 		histbb.append(b)
	# 		histgg.append(g)
	# 		histrr.append(r)
		
	# 	# histB = np.concatenate(histbb)
	# 	# histG = np.concatenate(histgg)
	# 	# histR = np.concatenate(histrr)
	# 	plt.plot(histbb, color='blue')
	# 	plt.plot(histgg, color='green')
	# 	plt.plot(histrr, color='red')
	# 	# plt.hist(histB, bins=25, alpha=0.5,histtype='step', label='blue',stacked=True, fill=False, color="blue")
	# 	# plt.hist(histG, bins=25, alpha=0.5,histtype='step', label='green',stacked=True, fill=False, color="green")
	# 	# plt.hist(histR, bins=25, alpha=0.5,histtype='step', label='red',stacked=True, fill=False, color="red")
	# 	# # Set labels and title
	# 	plt.xlabel('Value')
	# 	plt.xlim([0, 256])
	# 	plt.legend()
	# 	plt.ylabel('Frequency')
	# 	plt.title('Histogram')
	# 	plt.show()

	if len(points) != 0:
		# daaa = cv2.contourArea(points)
		points = np.array(points, dtype=np.int32)
		# Find the convex hull of the contour points
		hull = cv2.convexHull(points)
		# Find the bounding rectangle of the convex hull
		x, y, w, h = cv2.boundingRect(hull)
		# Draw the rectangle
		cv2.putText(frame, "h "+str(h), (x, int((y + h)/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
		cv2.putText(frame, "w "+str(w), (int((x + w)/2), y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
		cv2.putText(frame, "Aria "+str(w * h), (x, y + h-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
		
	# Afisarea fps-urilor pe imagine
	cv2.putText(frame, "FPS: {}".format(1.0 / (time.time() - start_time)), (10, 20),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

	cv2.putText(frame, "Dispersia: {}".format(variance), (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
	print("FPS: ", 1.0 / (time.time() - start_time))
	# show the frame and record if the user presses a key
	out.write(frame)
	cv2.imshow("rezultat final", frame)
	cv2.imshow("Thresh", thresh)
	cv2.imshow("Frame Delta", frameDelta)
	cv2.imshow("color_space",threshCheckColor)

	key = cv2.waitKey(1) & 0xFF
	if key == ord('q'):
		break
 
# cleanup the camera and close any open windows

cap.release()
out.release()
cv2.destroyAllWindows()