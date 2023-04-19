import cv2

for i in range(297,702):
    
    try:
        fileName = (f"Units/IMG_0{i}.JPG")
        img = cv2.imread(fileName)

        resized = cv2.resize(img, (640,640), interpolation=cv2.INTER_AREA)

            # # cv2.imshow("Resized image", resized)
            # # cv2.waitKey(0)
        cv2.imwrite(f"labeledUnits/IMG_0{i}.png", resized)
    except:
        print(i)
