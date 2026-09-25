"""Versioned spatial observations, association, reset and smoothing."""
from pathlib import Path
import numpy as np
from .spatial import gradient_saliency_center, place_crop

class SpatialPath:
    """Largest legal crop; face mode is a face detector, not generic subject understanding."""
    def __init__(self, mode, ratio, detector_path=None, alpha=.25):
        valid={"center","saliency","subject_proxy","subject_proxy_smooth","true_face","true_face_smooth"}
        if mode not in valid:raise ValueError(f"mode must be {valid}")
        self.mode,self.ratio,self.alpha=mode,ratio,alpha
        self.previous_image=None;self.center=None;self.face=None;self.detector=None
        self.observations=0;self.detections=0;self.resets=0
        if mode.startswith("true_face"):
            import cv2
            if not detector_path or not Path(detector_path).is_file():raise ValueError("face weights required")
            self.detector=cv2.FaceDetectorYN.create(str(detector_path),"",(320,320),.8,.3,5000)
    def step(self,rgb):
        import cv2
        h,w=rgb.shape[:2];thumb=cv2.resize(rgb,(64,36)).astype(np.float32)/255
        reset=self.previous_image is None or float(np.abs(thumb-self.previous_image).mean())>.25
        self.previous_image=thumb
        if reset:self.center=None;self.face=None;self.resets+=1
        scale=min(1,320/w);small=cv2.resize(rgb,(int(round(w*scale)),int(round(h*scale))))
        cx,cy,conf=gradient_saliency_center(small)
        if self.mode=="center":point=np.array([.5,.5])
        elif self.mode=="saliency":point=np.array([cx,cy])
        else:
            weight=np.clip((conf-.08)/.92,0,1)
            point=np.array([.5+weight*(cx-.5),.5+weight*(cy-.5)])
        if self.detector is not None:
            self.detector.setInputSize((small.shape[1],small.shape[0]))
            faces=self.detector.detect(cv2.cvtColor(small,cv2.COLOR_RGB2BGR))[1]
            if faces is not None and len(faces):
                centers=(faces[:,:2]+faces[:,2:4]/2)/np.array([small.shape[1],small.shape[0]])
                importance=faces[:,2]*faces[:,3]*faces[:,-1]
                if self.face is not None:
                    importance=importance/(1+8*np.linalg.norm(centers-self.face,axis=1))
                point=centers[int(np.argmax(importance))];self.face=point;self.detections+=1
        if self.mode.endswith("_smooth") and self.center is not None:
            point=self.alpha*point+(1-self.alpha)*self.center
        self.center=np.clip(point,0,1);self.observations+=1
        return place_crop(w,h,self.ratio,self.center[0]*w,self.center[1]*h),reset
