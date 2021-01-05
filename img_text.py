import math
import requests, json, random
import numpy as np
import cv2

def resize(img, scale:float = 0.5, px:int = -1):
  if px >= 0: scale = px / max(img.shape[0:2])
  if px > 1: return cv2.resize(img.copy(), (int(img.shape[1] * scale), int(img.shape[0] * scale)), interpolation = cv2.INTER_AREA)
  else: return cv2.resize(img.copy(), (1, 1), interpolation = cv2.INTER_AREA)

def black_border(img, scale:float = 0.09, px:int = -1):
  if px < 0: px = int(max(img.shape[0:2]) * scale)
  img = img.copy()
  black = (0,0,0) + (255,) * (img.shape[2] - 3)
  img[0:px,], img[-px:,], img[:,0:px], img[:,-px:] = (black,) * 4
  return img

def get_edgy_point(img, radius:int = 0, n:int = 1):
  n = max(1, n); radius = radius * 2 + 1; img = img.copy()
  scale = 1000 / max(img.shape[0:2])
  
  gray = resize(black_border(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if isinstance(img[0,0], np.ndarray) else img), px=1000)
  g_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
  g_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
  sorbel = cv2.addWeighted(g_x, 0.5, g_y, 0, 5, 0)
  proc_img = cv2.GaussianBlur(sorbel, (radius, radius), 0)
  

  if n > 1: # If n > 1, find the next brightest point and make it dark.
    pt = get_edgy_point(sorbel, radius=radius, n=n-1)
    cv2.rectangle(proc_img, pt1=(int(pt[0]-radius*2/scale), int(pt[1]-radius*2/scale)), pt2=(int(pt[0]+radius*2/scale), int(pt[1]+radius*2/scale)), color=(0,0,0), thickness=-1)
  
  (_, _, _, maxLoc) = cv2.minMaxLoc(proc_img)
  return tuple([int(x/scale) for x in maxLoc])

def get_brightest_point(img, radius:int = 10, n:int = 1):
  n = max(1, n); radius = radius * 2 + 1; img = img.copy()
  scale = 1000 / max(img.shape[0:2])
  
  if n > 1: # If n > 1, find the next brightest point and make it dark.
    pt = get_brightest_point(img, radius=radius, n=n-1)
    img = cv2.rectangle(img, pt1=(int(pt[0]-radius*10/scale), int(pt[1]-radius*10/scale)), pt2=(int(pt[0]+radius*10/scale), int(pt[1]+radius*10/scale)), color=(0,0,0), thickness=-1)
  
  proc_img = cv2.GaussianBlur(cv2.cvtColor(black_border(resize(img, px=1000)), cv2.COLOR_BGR2GRAY), (radius, radius), 0)

  (_, _, _, maxLoc) = cv2.minMaxLoc(proc_img)
  return tuple([int(x/scale) for x in maxLoc])

# https://gist.github.com/clungzta/b4bbb3e2aa0490b0cfcbc042184b0b4e
def alpha_overlay(bg, overlay, x=0, y=0): 
  b,g,r,a = cv2.split(overlay)
  overlay_color = cv2.merge((b,g,r))
  
  mask = cv2.medianBlur(a, int(int(max(bg.shape[0:2])/1000)/2)+1)

  h, w, _ = overlay_color.shape
  roi = bg[y:y+h, x:x+w]

  mod_bg = cv2.bitwise_and(roi.copy(),roi.copy(), mask = cv2.bitwise_not(mask))
  mod_fg = cv2.bitwise_and(overlay_color,overlay_color, mask = mask)
  bg[y:y+h, x:x+w] = cv2.add(mod_bg, mod_fg)
  return bg

def text_connect(img, text:str, pt1, pt2, color = (255, 255, 255), font = cv2.FONT_HERSHEY_SIMPLEX, size:float = 2, stroke:float = 5, force_90:bool = False,):
  matte = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
  txt_scalar = sum(img.shape[0:2])/1000
  mid_point = tuple([int((pt1[i] + pt2[i])/2) for i in range(2)])
  angle = math.degrees(math.atan2(pt2[0]-pt1[0], pt2[1]-pt1[1])) + 90;
  if angle > 90: angle += 180 # maximum readability
  if force_90: angle = int(angle/90) * 90

  # Calculate size to center text
  text_size, _ = cv2.getTextSize(text, font, int(size * txt_scalar), int(stroke * txt_scalar))
  text_origin = int(mid_point[0] - text_size[0] / 2), int(mid_point[1] + text_size[1] / 2)

  # Place text
  cv2.putText(matte, text, text_origin, cv2.FONT_HERSHEY_SIMPLEX, int(size * txt_scalar), color + (255,), int(stroke * txt_scalar))
  
  # Rotate text
  matrix = cv2.getRotationMatrix2D(mid_point, angle, 1)
  matte = cv2.warpAffine(matte, matrix, (matte.shape[1], matte.shape[0]))

  return alpha_overlay(img, matte)

def demo_text(img, text, force_90:bool = False, n_start:int = 0, color:tuple = (255, 255, 255), size:float = 2, stroke:float = 5):
  img = img.copy()
  pt1 = get_brightest_point(img, n=n_start+1)
  pt2 = get_brightest_point(img, n=n_start+2)
  text_connect(img, text, pt1, pt2, color=(0,)*3, size=size, stroke=stroke*2, force_90 = force_90)
  text_connect(img, text, pt1, pt2, color=color, size=size, stroke=stroke, force_90 = force_90)
  return img

def img_url_from_query(query, key):
  req = requests.get('https://pixabay.com/api/?'+
                      'key='+key+
                      '&q='+str(query).replace(' ','+')+
                      #basic safety settings
                      '&image_type=photo&orientation=horizontal&min_height=700&safesearch=true&'+
                      #try and get the most liked results
                      '&page=1&per_page=10&order=popular')
  req.raise_for_status(); req = req.json()
  return random.choice(req['hits'])['largeImageURL']

def cv_img_from_url(url):
  r = requests.get(url, stream=True).raw
  image = np.asarray(bytearray(r.read()), dtype="uint8")
  return cv2.imdecode(image, cv2.IMREAD_COLOR)

def cv_img_contrast(img, contrast, brightness):
  img = img.copy()
  img = cv2.addWeighted(img, contrast, img, 0, brightness)
  return img