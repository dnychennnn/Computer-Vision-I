import cv2
import numpy as np

NUM_IMAGES=14
image_prefix = "../images/frame"
image_suffix = ".png"

def main():
    objp = np.zeros((7*10,3), np.float32)
    objp[:,:2] = np.mgrid[0:7,0:10].T.reshape(-1,2)
    objpts=[]
    imgpts=[]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    imgs = []
    grays = []
    for i in range(NUM_IMAGES):
        img_path = image_prefix+"{0:0=3d}".format(i)+image_suffix
        img = cv2.imread(img_path)
        imgs.append(img.copy())
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        grays.append(gray)
    # task 1: call function
        ret, corners = cv2.findChessboardCorners(gray, (7,10))
        if ret == True:
            objpts.append(objp)
            corners2 = cv2.cornerSubPix(gray ,corners,(11,11),(-1,-1),criteria)
            imgpts.append(corners2)
            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (7,10), corners2,ret)
            cv2.imshow('img',img)
            cv2.waitKey(0)

    # task 2: call function
    _, cameramtx, distmtx, rotationmtx, translationmtx = cv2.calibrateCamera(objpts, imgpts, gray.shape, (3,3), None)
    print("Camera Matrix", cameramtx.astype(np.double), "\n",
        "Distortion Matrix", distmtx, "\n",
        "Rotation Matrix", rotationmtx, "\n",
        "Translation Matrix", translationmtx)
    # task 3: call function
    err =0
    for i in range(len(objpts)):
        reprojected_pts, _ = cv2.projectPoints(objpts[i], rotationmtx[i], translationmtx[i], cameramtx, distmtx)
        err = err + cv2.norm(imgpts[i],reprojected_pts, cv2.NORM_L2)/len(reprojected_pts)
        img = imgs[i]
        rep = reprojected_pts[:,0,:]
        orig = imgpts[0][:,0,:]
        for idx, r in enumerate(rep):
            ny = r[1]
            nx = r[0]
            oy = orig[idx, 1]
            ox = orig[idx, 0]
            cv2.circle(img, (nx,ny), 3, (0,0,255), -1)
            cv2.circle(img, (ox,oy), 2, (0,255,0), -1)

        cv2.imshow('reprojected points(red)',img)
        cv2.waitKey(0)
                # reprojected_pts = np.expand_dims(np.array(reprojected_pts)[:, 0, :], axis=0)
    print("total error: ", err/len(objpts))

    
        

    # task 4: call function
    def differenceImageV6(img1, img2):
        a = img1-img2
        b = np.uint8(img1<img2) * 254 + 1
        return a * b
    for compenstate_img in grays:
        h, w = compenstate_img.shape
        newcameramtx, roi=cv2.getOptimalNewCameraMatrix(cameramtx,distmtx,(w,h),1,(w,h))
        dst = cv2.undistort(compenstate_img, cameramtx, distmtx, newcameramtx)
        print("Absolute Difference", np.mean(differenceImageV6(dst, compenstate_img)))
        # crop the image
        x,y,w,h = roi
        dst = dst[y:y+h, x:x+w]
        # cv2.imshow("orig", img)
        cv2.imshow("undistort", dst)
        cv2.waitKey(0)

    # task 5: call function

    print("FINISH!")

main()