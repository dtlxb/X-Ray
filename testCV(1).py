
# coding: utf-8

# In[ ]:





# In[1]:


import sys
print(sys.version)


# In[2]:


import cv2 as cv2
import numpy as np
import math


# In[3]:


print(cv2.__version__)


# In[4]:


def cv_imread(filePath):  
    ##cv_img=cv2.imdecode(np.fromfile(filePath,dtype=np.uint8),-1)  
    ## imdecode读取的是rgb，如果后续需要opencv处理的话，需要转换成bgr，转换后图片颜色会变化  
    ##cv_img=cv2.cvtColor(cv_img,cv2.COLOR_RGB2BGR)  
    img=cv2.imread(path,1)  
    return img 


# In[5]:


#path2='D:\\Out\\AbnormalOut\\IMG-0001-000014.jpg' 

#path2='D:\\Out\\AbnormalOut\\IMG-0003-000017.jpg' 
#path='D:\\Out\\AbnormalOut\\317cut.png' 
path='D:\\Out\\NormalOut\\IMG-0001-0000119.png' 
path='D:\\Out\\NormalOut\\IMG-0001-00001.png'
path='D:\\Out\\NormalOut\\IMG-0001-000012.png'
#path='D:\\Out\\NormalOut\\IMG-0001-000013.png'
#path='D:\\Out\\NormalOut\\IMG-0001-000014.png'
path='D:\\Out\\NormalOut\\IMG-0001-000015.png'
#path='D:\\Out\\NormalOut\\IMG-0001-000018.png'
#path='D:\\Out\\AbnormalOut\\IMG-0001-000010.png' 
path='D:\\Out\\AbnormalOut\\IMG-0001-0000140.jpg' 
#path='D:\\Out\\AbnormalOut\\317cut.png' 
#path='D:\\Out\\result.png'
#path='D:\\Out\\Origin\\IMG-0001-000012.jpg'

# feature compare example
path2 = 'D:\\Out\\Examples\\01.png' 
#path2 = 'D:\\Out\\Examples\\02.png' 
#path2 = 'D:\\Out\\Origin\\IMG-0001-000013.jpg'


img=cv2.imread(path,1)  
img0=cv2.imread(path,1)
img1=cv2.imread(path2,1)


# In[6]:


img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#img = cv2.equalizeHist(img)
#img = cv2.GaussianBlur(img,(5,5),0)
img = cv2.medianBlur(img,3)

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

#img = cv2.equalizeHist(img)
#img = cv2.GaussianBlur(img,(5,5),0)
img1 = cv2.medianBlur(img1,3)


# In[7]:


# 图像的左右-内外信息：
# 外侧在左，为0，外侧在右，为1
lateral_is_on_the_right = 0


# In[8]:


# compare features
sift = cv2.xfeatures2d.SIFT_create()
#sift = cv2.SURF()
 
kp1, des1 = sift.detectAndCompute(img, None)
kp2, des2 = sift.detectAndCompute(img1, None)

#print(kp1)
for i in kp2:
    print(i.pt)
    
print('------')
for i in kp1:
    print(i.pt)

bf = cv2.BFMatcher()
#返回k个最佳匹配
matches = bf.knnMatch(des2, des1, k=2)#这里的顺序要与drawMatch中一致
good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append([m])
#img3 = cv2.drawMatchesKnn(img, kp1, img1, kp2, good[:10], None, flags=2)
img3 = cv2.drawMatchesKnn(img1, kp2, img, kp1, good[:10], None, flags=2)

print(len(good))

#for i in kp1:
#    i0 = (int(i.pt[0]),int(i.pt[1]))
#    img3 = cv2.drawMarker(img3, i0,(0,0,255),cv2.MARKER_STAR,20,1, 8 )


# In[9]:


# 输出匹配点信息。不要运行
for i in good:
    print(i[0].queryIdx)#queryIdx是第一个参数图片中的kp的Idx，train是第二个。
    print(i[0].trainIdx)#所以在这里，query是样图（path2，img1），train是要分析的图片（path，img）。
    print(i[0].imgIdx)

examplePoints = []
targetPoints = []

for i in good:
    examplePoints.append(kp2[i[0].queryIdx])
    targetPoints.append(kp1[i[0].trainIdx])

print('第一组匹配点')
print(examplePoints[0].pt)
print(targetPoints[0].pt)
print('第二组匹配点')
print(examplePoints[1].pt)
print(targetPoints[1].pt)


# In[312]:


# 测试：标注一个keypoint
i = kp2[20]
i0 = (int(i.pt[0]),int(i.pt[1]))
img3 = cv2.drawMarker(img3, i0,(0,0,255),cv2.MARKER_STAR,20,1, 8 )


# In[10]:


# 显示keypoint对照图
cv2.namedWindow('image',cv2.WINDOW_NORMAL)  
cv2.imshow('image',img3)  
k=cv2.waitKey(0) 


# In[60]:


# 如果直接处理边缘，这里开始就要跳过。
# 否则，从这里开始插入二值化
th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C , cv2.THRESH_BINARY,31,2)
ret , white_th = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
kernel = np.ones((3,3),np.uint8)
ret , black_th = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
black_th = cv2.dilate(black_th, kernel,iterations=10)
th3 = th3+white_th
th3 = th3-(255-black_th)
kernel = np.ones((5,5),np.uint8)
th3 = cv2.morphologyEx(th3,cv2.MORPH_CLOSE,kernel)
th3 = cv2.erode(th3,kernel,iterations = 1)
th3 = cv2.morphologyEx(th3,cv2.MORPH_CLOSE,kernel)
kernel = np.ones((3,3),np.uint8)
th3 = cv2.dilate(th3,kernel,iterations = 2)
#img = th3


# In[11]:


# 跳到这里，进行轮廓预处理
img = cv2.medianBlur(img,5)


# In[12]:


# 得到轮廓
edges = cv2.Canny(img,20,90)  # 有二值化就是th3，否则就是Img
#edges = cv2.Canny(img,20,80)  # 有二值化就是th3，否则就是Img
cv2.namedWindow('image',cv2.WINDOW_NORMAL)  
cv2.imshow('image',edges)  
k=cv2.waitKey(0) 


# In[13]:


# 轮廓处理，去掉过短的轮廓
none,contours,hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
saveCs = []
for i in range(0,len(contours)):
    #print(cv2.arcLength(contours[i],False))
    if cv2.arcLength(contours[i],False)>100:
        saveCs.append(contours[i])
#print(len(saveCs))
imgcp = img.copy()
imgcp = cv2.drawContours(imgcp,saveCs,-1,(0,255,0),1)


# In[14]:


# 输出保留的轮廓的条数
print(len(saveCs))


# In[15]:


# 定义计算数据要使用的数学函数

# 计算点d0到直线d1d2的距离
def calculate_feet(d0,d1,d2):
    x1 = d1[0]
    y1 = d1[1]
    x2 = d2[0]
    y2 = d2[1]
    x0 = d0[0]
    y0 = d0[1]
    
    k = ((x1 - x0) * (x2 - x1) + (y1 - y0) * (y2 - y1)) / ((x2 - x1)**2 + (y2 - y1)**2)
    k = -k
    feet0 = (k*(x2-x1) + x1, k*(y2-y1) + y1)
    feet = (int(feet0[0]),int(feet0[1]))
    return feet

# 计算d0和d1的两点间距
def distance(d0, d1):
    return math.sqrt((d0[0] - d1[0])**2 + (d0[1] - d1[1])**2)


# In[48]:


# 定义用于手动标定的鼠标响应函数
imgcp = img.copy()
imgcp = cv2.drawContours(imgcp,saveCs,-1,(0,255,0),1)

# 手动输入六个关键点
#bottom_left, bottom_right, middle_left, middle_right, up_left, up_right = 0
data = []

def calculate():
    # 六个关键点
    bottom_left = data[4]
    bottom_right = data[5] 
    middle_left = data[2] 
    middle_right = data[3] 
    up_left = data[0]
    up_right = data[1]
    
    # 计算数值
    # 可视化
    cv2.line(imgcp,up_left,up_right,(255,0,0),1)
    cv2.line(imgcp,bottom_left,bottom_right,(0,0,255),1)

    # 计算高度
    lateral = middle_right
    up_lateral = up_right
#     x1 = bottom_left[0]
#     y1 = bottom_left[1]
#     x2 = bottom_right[0]
#     y2 = bottom_right[1]
#     x0 = lateral[0]
#     y0 = lateral[1]
    
#     k = ((x1 - x0) * (x2 - x1) + (y1 - y0) * (y2 - y1)) / ((x2 - x1)**2 + (y2 - y1)**2)
#     k = -k
#     lateral_feet0 = (k*(x2-x1) + x1, k*(y2-y1) + y1)
#     lateral_feet = (int(lateral_feet0[0]),int(lateral_feet0[1]))
    lateral_feet = calculate_feet(lateral,bottom_left, bottom_right)
    print(distance(lateral_feet, lateral))
    up_feet = calculate_feet(up_lateral,bottom_left, bottom_right)
    print(distance(up_lateral, up_feet))
    
    cv2.line(imgcp, lateral, lateral_feet,(0,255,0),3)
    cv2.line(imgcp, up_lateral, up_feet,(255,0,0),3)
    cv2.imshow('image',imgcp)
    
    
# mouse callback
def click(event,x,y,flags,param):
    global data
    if event == cv2.EVENT_LBUTTONDOWN:
        data.append((x,y))
        print(data)
        if (len(data) >= 6):
            calculate();
    if event == cv2.EVENT_RBUTTONDOWN:
        data = []



# In[49]:


# imgcp：原图+轮廓

cv2.namedWindow('image',cv2.WINDOW_NORMAL)  
cv2.setMouseCallback('image',click)
cv2.imshow('image',imgcp)  
k=cv2.waitKey(0)  


# In[68]:


# 迭代1：找轮廓，只显示外上侧轮廓和外上侧基准点。
#print(saveCs[1])

# 判断特征点
# 这两个函数还没有实现，用后面的手动代码代替。
#kpt_middle_target = find_middle(good)
#kpt_up_lateral_target = find_up_lateral(good)

# 样例图中心特征点的下标。
idx_middle_example = 20

# 遍历匹配成功的特征点对
middle = -1
up_lateral = -1
for i in good:
    print(i[0].queryIdx)
    print(i[0].trainIdx)
    if (i[0].queryIdx == idx_middle_example):# 找到了关节中心的特征点
        print('found middle!')
        middle = i[0].trainIdx
    if (i[0].queryIdx == 1):
        up_lateral = i[0].trainIdx
        
print(middle)
#print(up_lateral)

# 在target图上标记被匹配为中心点的特征点
kpt_middle_target = (int(kp1[middle].pt[0]),int(kp1[middle].pt[1]))
kpt_up_lateral_target = (int(kp1[up_lateral].pt[0]),int(kp1[up_lateral].pt[1]))
imgcp = cv2.drawMarker(imgcp, kpt_middle_target,(0,0,255),cv2.MARKER_STAR,20,1, 8 )
#imgcp = cv2.drawMarker(imgcp, kpt_up_lateral_target,(0,0,255),cv2.MARKER_STAR,20,1, 8 )

# 寻找轮廓


# In[69]:


# 目标图中标注中心特征点
cv2.namedWindow('image',cv2.WINDOW_NORMAL)  
cv2.imshow('image',imgcp)  
k=cv2.waitKey(0) 


# In[70]:


# 根据中心特征点找轮廓。 p.s. 可以先放弃这个算法了。跳过此cell
# 中心特征点：kpt_middle_target
# 轮廓集：saveCs
print(kpt_middle_target)
print(len(saveCs))
#print(saveCs[0])

# 经过中心点正上方的最近轮廓
def up_nearest_contour(kpt_middle_target, saveCs):
    x = kpt_middle_target[0]
    y = kpt_middle_target[1]
    min_y = 99999999
    pt_reserve = []
    contour_reserve = []
    counter = 0
    contour_idx = -1
    # find x1 = x && y1 < y
    for contour in saveCs:
        for pt in contour:
            if ((pt[0][0] == x)&((y - pt[0][1])<min_y)&((y - pt[0][1])>0)):
                pt_reserve = pt
                contour_reserve = contour
                contour_idx = counter
                min_y = y - pt[0][1]
        counter+=1
                
    #print(pt_reserve)
    #print(contour_reserve)
    return contour_idx

# 找上侧轮廓
up_contour_res = up_nearest_contour(kpt_middle_target,saveCs)
print(up_contour_res)

# 取出上侧轮廓
up_contour = []
if (up_contour_res != -1):
    up_contour = [saveCs[up_contour_res]]


# In[71]:


# 找上方轮廓version1：只找一条最大的
# 被下文的更完整的算法取代。跳过此cell。

# 没有中心点，直接找轮廓
def find_max_X_distance(saveCs):
    max = 0
    for contour in saveCs:
        max_X = 0
        min_X = 99999999
        for pt in contour:
            if (pt[0][0] > max_X):
                max_X = pt[0][0]
            if (pt[0][0] < min_X):
                min_X = pt[0][0]
        imax = max_X - min_X
        if (imax > max):
            max = imax
    print('max X distance:')
    print(max)
    return max

def get_Xs_with_threshold(saveCs, X_threshold):
    counter = 0
    list = []
    for contour in saveCs:
        max_X = 0
        min_X = 99999999
        for pt in contour:
            if (pt[0][0] > max_X):
                max_X = pt[0][0]
            if (pt[0][0] < min_X):
                min_X = pt[0][0]
        imax = max_X - min_X
        if (imax >= X_threshold):
            list.append(counter)
        counter+=1
    print(list)
    return list

def upmost(X_list):
    min_Y = 99999999
    min_index = -1
    for i in X_list:
        contour = saveCs[i]
        counter = 0
        sum = 0
        for pt in contour:
            sum += pt[0][1]
            counter += 1
        avg_Y = sum/counter
        if (avg_Y < min_Y):
            min_Y = avg_Y
            min_index = i
    print(min_index)
    return min_index

def brute_find_up_contour(saveCs):
    max_X_distance = find_max_X_distance(saveCs)
    X_threshold = int(0.5*max_X_distance)
    X_list = get_Xs_with_threshold(saveCs,X_threshold)
    res = upmost(X_list)
    return res

if (up_contour_res == -1):
    up_contour_res = brute_find_up_contour(saveCs)

up_contour_res = brute_find_up_contour(saveCs)
print(up_contour_res)


# In[50]:


# 找上方轮廓version2：
# 若轮廓断裂，则找到多条子轮廓，并作判断

# 没有中心点，直接找轮廓
def find_max_X_distance(saveCs):
    max = 0
    for contour in saveCs:
        max_X = 0
        min_X = 99999999
        for pt in contour:
            if (pt[0][0] > max_X):
                max_X = pt[0][0]
            if (pt[0][0] < min_X):
                min_X = pt[0][0]
        imax = max_X - min_X
        if (imax > max):
            max = imax
    print('max X distance:')
    print(max)
    return max

def get_Xs_with_threshold(saveCs, X_threshold):
    counter = 0
    list = []
    for contour in saveCs:
        max_X = 0
        min_X = 99999999
        for pt in contour:
            if (pt[0][0] > max_X):
                max_X = pt[0][0]
            if (pt[0][0] < min_X):
                min_X = pt[0][0]
        imax = max_X - min_X
        if (imax >= X_threshold):
            list.append(counter)
        counter+=1
    #print(list)
    return list

def upmost(X_list):
    min_Y = 99999999
    min_index = -1
    for i in X_list:
        contour = saveCs[i]
        counter = 0
        sum = 0
        for pt in contour:
            sum += pt[0][1]
            counter += 1
        avg_Y = sum/counter
        if (avg_Y < min_Y):
            min_Y = avg_Y
            min_index = i
    #print(min_index)
    return min_index

def brute_find_up_contour(saveCs):
    max_X_distance = find_max_X_distance(saveCs)
    X_threshold = int(0.5*max_X_distance)
    X_list = get_Xs_with_threshold(saveCs,X_threshold)
    res = X_list
    return res

up_contour_res = brute_find_up_contour(saveCs)
print(up_contour_res)

# 所有疑似上方轮廓的较长子轮廓
sub_up_contours = []
for i in up_contour_res:
    sub_up_contours.append(saveCs[i])
print(len(sub_up_contours))


# In[51]:


# 去掉X方向上重叠且不是最高的轮廓
SUC_overlap_threshold = 0.1
# 重叠部分的比重在这个值以下，会被认为没有重叠

def SUC_is_beneath(down, up):
    # receive 2 contours, return bool if down overlaps and beneathes up
    # step 1: check overlap
    # 左右界
    dl = 99999999
    ul = 99999999
    dr = 0
    ur = 0
    for i in down:
        if (i[0][0] > dr):
            dr = i[0][0]
        if (i[0][0] < dl):
            dl = i[0][0]
    for i in up:
        if (i[0][0] > ur):
            ur = i[0][0]
        if (i[0][0] < ul):
            ul = i[0][0]
            
    # 重叠关系

    if (dr<=ul) or (dl>=ur):
        return False #they don't overlap
    
    lap_length = -1
    if (dl<=ul):
        if (dr<=ur):
            lap_length = dr-ul
        else:
            lap_length = ur-ul
    else:
        if (dr<=ur):
            lap_length = dr-dl
        else:
            lap_length = ur-dl
    # 没有大部分重叠，返回false。否则，认为重叠了，再进行上下检查。
    #print(lap_length)
    if (lap_length <= SUC_overlap_threshold * (dr-dl)) and (lap_length <= SUC_overlap_threshold * (ur-ul)):
        return False
    
    #print(lap_length)
    # step 2: check down-up
    u_sum = 0
    d_sum = 0
    for i in up:
        u_sum += i[0][1]
    for i in down:
        d_sum += i[0][1]
    u_sum /= len(up)
    d_sum /= len(down)
    if (u_sum < d_sum):
        return True
    else:
        # 相等的情况包含了同一条轮廓与自己比较的情况
        return False
    


# In[52]:


# 轮廓（们）在sub_up_contours里
# 若不止一条，则去重，然后合并成一条
up_contour_final = []

if (len(sub_up_contours) > 1):
    
    sub_up_contour_res = []
    
    # 子轮廓去重叠
    SUC_del_list = []
    for i in range(0,len(sub_up_contours)):
        for another in sub_up_contours:
            print(SUC_is_beneath(sub_up_contours[i],another))
            if SUC_is_beneath(sub_up_contours[i],another):
                SUC_del_list.append(i)        
    
    # 把所有未删掉的留着
    for i in range(0,len(sub_up_contours)):
        if (i not in SUC_del_list):
            sub_up_contour_res.append(sub_up_contours[i])
    
    # 把res中的所有轮廓合并成一条，存入final
    
    # 初始化字典
    dic_RC_merge={} # X-Y，只存储最大（即最下端）的Y值
    for i in range(0,imgcp.shape[:2][1]):
        dic_RC_merge[i] = 0 
    
    # 向字典中存入轮廓的坐标
    for contour in sub_up_contour_res:
        for i in contour:
            if (i[0][1] > dic_RC_merge[i[0][0]]):
                dic_RC_merge[i[0][0]] = i[0][1]
    
    # 填满字典，使其成为一条合理的轮廓
    # 扫两遍，正反各一遍。
    # 两遍之后，再扫一遍填入所有值。
    # 初始化weights
    RCM_weights = {}
    for i in range(0,imgcp.shape[:2][1]):
        RCM_weights[i] = 0
    # 正
    RCM_counter = 0
    RCM_recorder = 0
    for i in range(0,imgcp.shape[:2][1]):
        if dic_RC_merge[i]==0:
            RCM_counter+=1
            dic_RC_merge[i] += 10*RCM_recorder/RCM_counter
            RCM_weights[i] += 10/RCM_counter
        else:
            RCM_counter = 0
            RCM_recorder = dic_RC_merge[i]
    # 反
    RCM_counter = 0
    RCM_recorder = 0
    for i in range(0,imgcp.shape[:2][1]):  # 0~max
        j = imgcp.shape[:2][1]-1-i         # max~0
        if dic_RC_merge[j]==0:
            RCM_counter+=1
            dic_RC_merge[j] += 10*RCM_recorder/RCM_counter
            RCM_weights[j] += 10/RCM_counter
        else:
            RCM_counter = 0
            RCM_recorder = dic_RC_merge[j]
    # 合并计算
    for i in range(0,imgcp.shape[:2][1]):
        if (RCM_weights[i] != 0):
            dic_RC_merge[i] = int(dic_RC_merge[i]/RCM_weights[i])
    
    #print([k,dic_RC_merge[k]])
    for k in dic_RC_merge:
        #print(k)
        up_contour_final.append([np.array([k,dic_RC_merge[k]])])
    print(up_contour_final)
    
else:
    up_contour_final = sub_up_contours[0]


# In[53]:


up_contour_final = np.array(up_contour_final)
print(type(up_contour_final))
print(up_contour_final)


# In[54]:


# 找到的上方轮廓！
imgcp = img.copy()

#print(saveCs[0])
# 从上方轮廓开始，重新画
imgcp = cv2.drawContours(imgcp,np.array([up_contour_final]),-1,(0,255,0),1)
#imgcp = cv2.drawContours(imgcp,[saveCs[0]],-1,(0,255,0),1)
cv2.namedWindow('image',cv2.WINDOW_NORMAL)  
cv2.imshow('image',imgcp)  
k=cv2.waitKey(0) 


# In[55]:


# 股骨髁底侧连线
def takeX(elem):
    return elem[0][0]
sorted_up_contour_res = sorted(up_contour_final,key=takeX)
#print(a[1])
#print(saveCs[0])
def avg(list):
    sum = 0
    for i in list:
        sum += i
    return sum/len(list)

# 局部最大值的比较区间
min_X = 99999999
max_X = -1
for i in sorted_up_contour_res:
    if (i[0][0] < min_X):
        min_X = i[0][0]
    if (i[0][0] > max_X):
        max_X = i[0][0]
print(min_X)
print(max_X)
quarterRange = int((max_X - min_X)//4)
print(quarterRange)

# 局部最大值
max_is = []
for i in sorted_up_contour_res:
    # 左右范围
    left = min_X if (i[0][0] - quarterRange < min_X) else (i[0][0] - quarterRange)
    right = max_X if (i[0][0] + quarterRange > max_X) else (i[0][0] + quarterRange)
    # 范围内最大
    max_Y = -1
    for j in sorted_up_contour_res:
        if ((j[0][0] >= left) and (j[0][0] <= right)):
            if (j[0][1] > max_Y):
                max_Y = j[0][1]
    # 最值点
    #print(i[0])
    if (max_Y == i[0][1]):
        max_is.append((i[0][0],i[0][1]))

# 合并相连的点
print(max_is)
max_is_idx = {}
max_is_res = {}
for i in max_is:
    max_is_idx[i[1]] = []
for i in max_is:
    max_is_idx[i[1]].append(i[0])
for i in max_is_idx:
    max_is_res[i] = int(avg(max_is_idx[i]))
print(max_is_res)

# 把上侧连线画在图上。
max_is_res_k = []
for m in max_is_res:
    max_is_res_k.append(m)
cv2.line(imgcp,(max_is_res[max_is_res_k[0]],max_is_res_k[0]),(max_is_res[max_is_res_k[1]],max_is_res_k[1]),(255,0,0),1)
for m in max_is_res:
    imgcp = cv2.drawMarker(imgcp, (max_is_res[m],m),(0,0,255),cv2.MARKER_STAR,20,1, 8 )
    
# 附加：记下上侧两个基准点，方便之后访问。
up_feature_left = ()
up_feature_right= ()
if (max_is_res[max_is_res_k[0]] > max_is_res[max_is_res_k[1]]):
    up_feature_left = (max_is_res[max_is_res_k[1]],max_is_res_k[1])
    up_feature_right = (max_is_res[max_is_res_k[0]],max_is_res_k[0])
else:
    up_feature_left = (max_is_res[max_is_res_k[0]],max_is_res_k[0])
    up_feature_right = (max_is_res[max_is_res_k[1]],max_is_res_k[1])
print(up_feature_left)


# In[56]:


cv2.namedWindow('image',cv2.WINDOW_NORMAL)  
cv2.imshow('image',imgcp)  
k=cv2.waitKey(0) 


# In[57]:


# 下侧棘突
# sorted_up_contour_res 上轮廓
# max_is_res[Y] = X 上轮廓双基准点
# saveCs 所有轮廓
# max_is_res_k 下标访问max_is_res
# up_feature_left,up_feature_right 股骨髁下端基准点

# 找所有局部最小值Y
def down_tops(saveCs, range_lm, list_local_minimal):
    for contour in saveCs:
        c_sorted = sorted(contour,key=takeX)
        # X范围
        min_X = 99999999
        max_X = -1
        for i in c_sorted:
            if (i[0][0] < min_X):
                min_X = i[0][0]
            if (i[0][0] > max_X):
                max_X = i[0][0]
        # 
        for i in c_sorted:
            # 去掉端点，否则一个上升的端点一定会被判定为最值
            if ((i[0][0] == min_X) or (i[0][0] == max_X)):
                continue
            
            left = min_X if (i[0][0] - range_lm < min_X) else (i[0][0] - range_lm)
            right = max_X if (i[0][0] + range_lm > max_X) else (i[0][0] + range_lm)
            # 范围内最大
            min_Y = 99999999
            for j in c_sorted:
                if ((j[0][0] >= left) and (j[0][0] <= right)):
                    if (j[0][1] < min_Y):
                        min_Y = j[0][1]
            # 最值点
            #print(i[0])
            if (min_Y == i[0][1]):
                list_local_minimal.append((i[0][0],i[0][1]))

list_local_minimal = []
range_lm = int(abs(max_is_res[max_is_res_k[0]] - max_is_res[max_is_res_k[1]])//10)#根据基准点间距确定最值范围
print(range_lm)
down_tops(saveCs, range_lm, list_local_minimal)
#print(list_local_minimal)

# 在图上画出所有最值点
#for m in list_local_minimal:
#    imgcp = cv2.drawMarker(imgcp, m,(0,0,255),cv2.MARKER_STAR,20,1, 8 )
    
# 为上侧轮廓建立X-最大Y字典
XY_SUCR = {}
for i in sorted_up_contour_res:
    if (i[0][0] not in XY_SUCR.keys()):
        XY_SUCR[i[0][0]] = i[0][1]
    else:
        if (XY_SUCR[i[0][0]] < i[0][1]):
            XY_SUCR[i[0][0]] = i[0][1]
# 补上空白部分
XY_SUCR_TMP = -1
for i in range(sorted_up_contour_res[0][0][0],sorted_up_contour_res[-1][0][0]):
    if (i not in XY_SUCR.keys()):# 第一次不会为空
        XY_SUCR[i] = XY_SUCR_TMP
    XY_SUCR_TMP = XY_SUCR[i] 
#print(XY_SUCR)

# 删除范围外的点
list_local_minimal_selected = []
for m in list_local_minimal:
    if ((m[0] > up_feature_left[0]) and (m[0] < up_feature_right[0])):
        if (m[1] > XY_SUCR[m[0]]):
            list_local_minimal_selected.append(m)
#print(list_local_minimal_selected)

# 找最小的两个Y，认为是两个棘突
down_tips = []
YX_LLMS = {}
for i in list_local_minimal_selected:
    YX_LLMS[i[1]] = []
for i in list_local_minimal_selected:
    YX_LLMS[i[1]].append(i[0])

minY_YX_LLMS = 99999999
for Y in YX_LLMS:
    if (Y < minY_YX_LLMS):
        minY_YX_LLMS = Y
down_tips.append((int(avg(YX_LLMS[minY_YX_LLMS])),minY_YX_LLMS))
del YX_LLMS[minY_YX_LLMS]

minY_YX_LLMS = 99999999
for Y in YX_LLMS:
    if (Y < minY_YX_LLMS):
        minY_YX_LLMS = Y
down_tips.append((int(avg(YX_LLMS[minY_YX_LLMS])),minY_YX_LLMS))
del YX_LLMS[minY_YX_LLMS]

print(down_tips)
# 把下侧棘突画在图上
for m in down_tips:
    imgcp = cv2.drawMarker(imgcp, m,(0,0,255),cv2.MARKER_STAR,20,1, 8 )

# resort down_tips
if (down_tips[0][0] > down_tips[1][0]):
    down_tips = [down_tips[1],down_tips[0]]


# In[58]:


cv2.namedWindow('image',cv2.WINDOW_NORMAL)  
cv2.imshow('image',imgcp)  
k=cv2.waitKey(0) 


# In[59]:


print(XY_SUCR)


# In[60]:


# 下侧平台连线
# sorted_up_contour_res 上轮廓
# saveCs 所有轮廓
# up_feature_left,up_feature_right 股骨髁下端基准点
# XY_SUCR 上侧轮廓的X-最大Y字典

# 在下侧平台之前：
# fill XY_SUCR in all possible X values
#print(imgcp.shape[:2])
#print(list(XY_SUCR.keys()))
for i in range(0,list(XY_SUCR.keys())[0]):
    XY_SUCR[i] = XY_SUCR[list(XY_SUCR.keys())[0]]
for i in range(list(XY_SUCR.keys())[-1]+1,imgcp.shape[:2][1]):
    XY_SUCR[i] = XY_SUCR[list(XY_SUCR.keys())[-1]]
print(XY_SUCR)


def find_down_lateral_left(saveCs, DLL_list):
    for contour in saveCs:
        # find its DLL_point
        min_DLL_val = 99999999
        DLL_res = (-1,-1)
        for i in contour:
            tmp_DLL_val = i[0][0]+i[0][1] # y=-x+b
            if (tmp_DLL_val < min_DLL_val):
                min_DLL_val = tmp_DLL_val
                DLL_res = (i[0][0], i[0][1])
        # check if it's under up_res
        if (DLL_res[1] <= XY_SUCR[DLL_res[0]]):
            continue
        else:
            DLL_list.append(DLL_res)
        
    # check all DLLs to find the min one 
    min_DLL = 99999999
    res = (-1,-1)
    for i in DLL_list:
        if (i[0]+i[1] <= min_DLL):
            min_DLL = i[0]+i[1]
            res = i
    
    return res # a_contour_with_max_DLL

def find_down_lateral_right(saveCs, DLR_list):
    for contour in saveCs:
        # find its DLL_point
        min_DLR_val = 99999999
        DLR_res = (-1,-1)
        for i in contour:
            tmp_DLR_val = i[0][1]-i[0][0] # y-x
            if (tmp_DLR_val < min_DLR_val):
                min_DLR_val = tmp_DLR_val
                DLR_res = (i[0][0], i[0][1])
        # check if it's under up_res
        if (DLR_res[1] <= XY_SUCR[DLR_res[0]]):
            continue
        else:
            DLR_list.append(DLR_res)
        
    # check all DLLs to find the min one 
    min_DLR = 99999999
    res = (-1,-1)
    for i in DLR_list:
        if (i[1]-i[0] < min_DLR):
            min_DLR = i[1]-i[0]
            res = i
    
    return res # a_contour_with_max_DLR

DLL_list = []
DLR_list = []
DLL = find_down_lateral_left(saveCs, DLL_list)
DLR = find_down_lateral_right(saveCs, DLR_list)
print(DLL_list)
print(DLR_list)
print(DLL)
print(DLR)
#for m in DLL_list:
#    imgcp = cv2.drawMarker(imgcp, m,(0,0,255),cv2.MARKER_STAR,20,1, 8 )
#imgcp = cv2.drawMarker(imgcp, DLL,(0,0,255),cv2.MARKER_STAR,20,1, 8 )
#imgcp = cv2.drawMarker(imgcp, DLR,(0,0,255),cv2.MARKER_STAR,20,1, 8 )


# In[61]:


# 找到DLL/DLR对应的轮廓
def find_point_in_contours(point, saveCs):
    for contour in saveCs:
        for i in contour:
            if ((i[0][0]==point[0])&(i[0][1]==point[1])):
                return contour

DLL_C = find_point_in_contours(DLL,saveCs)
DLR_C = find_point_in_contours(DLR,saveCs)

DL_universal_shift_threshold = 0.5 # 判断倾斜的参数。可以理解为tan值

def check_left_incline(index, c_sorted):
    # 检查右下边缘是否向左上方倾斜
    left_bound = index-10 if index > 10 else 0
    for i in range(left_bound, index):
        x_diff = c_sorted[index][0][0] - c_sorted[i][0][0]
        y_diff = c_sorted[index][0][1] - c_sorted[i][0][1]
        if (y_diff > DL_universal_shift_threshold*x_diff): #不仅向上斜，而且还足够斜
            return True  
    return False

def check_right_incline(index, c_sorted):
    # 检查右下边缘是否向左上方倾斜
    if (index == len(c_sorted)):
        return
    right_bound = index+11 if index +11 < len(c_sorted) else len(c_sorted) # 这句有bug吗？
    for i in range(index+1,right_bound):
        x_diff = c_sorted[i][0][0] - c_sorted[index][0][0]
        y_diff = c_sorted[index][0][1] - c_sorted[i][0][1]
        if (y_diff > DL_universal_shift_threshold*x_diff): #不仅向上斜，而且还足够斜
            return True      
    return False

def DLL_shift(DLL,DLL_C):
    c_sorted = sorted(DLL_C,key=takeX)
    index = 0
    
    # find index of DLR in contour
    while (index < len(c_sorted)):
        if ((c_sorted[index][0][0] == DLL[0]) & (c_sorted[index][0][1] == DLL[1])):
            break
        index += 1
    
    # shift along contour
    right_incline = check_right_incline(index, c_sorted)
    while (right_incline):
        print('right shifted!')
        index += 1
        right_incline = check_right_incline(index, c_sorted)
        
    return (c_sorted[index][0][0],c_sorted[index][0][1])


def DLR_shift(DLR,DLR_C):
    c_sorted = sorted(DLR_C,key=takeX)
    index = 0
    
    # find index of DLR in contour
    while (index < len(c_sorted)):
        if ((c_sorted[index][0][0] == DLR[0]) & (c_sorted[index][0][1] == DLR[1])):
            break
        index += 1
    
    # shift along contour
    left_incline = check_left_incline(index, c_sorted)
    while (left_incline):
        print('left shifted!')
        index -= 1
        left_incline = check_left_incline(index, c_sorted)
        
    return (c_sorted[index][0][0],c_sorted[index][0][1])

# 在边缘上移动，找到最佳点
DLL_shifted = DLL_shift(DLL,DLL_C)
DLR_shifted = DLR_shift(DLR,DLR_C)

# 点-连线-原轮廓
imgcp = cv2.drawMarker(imgcp, DLR_shifted,(0,0,255),cv2.MARKER_STAR,20,1, 8 )
imgcp = cv2.drawMarker(imgcp, DLL_shifted,(0,0,255),cv2.MARKER_STAR,20,1, 8 )
cv2.line(imgcp,DLL_shifted,DLR_shifted,(255,0,0),1)
imgcp = cv2.drawContours(imgcp,[DLL_C,DLR_C],-1,(0,255,0),1)


# In[ ]:


cv2.namedWindow('image',cv2.WINDOW_NORMAL)  
cv2.imshow('image',imgcp)  
k=cv2.waitKey(0) 


# In[63]:


if (not lateral_is_on_the_right):
    distance_lateral = distance(up_feature_left, calculate_feet(up_feature_left, DLL,DLR))
    height_tip = distance(down_tips[0], calculate_feet(down_tips[0], DLL,DLR))
    print('外侧膝关节间缝隙宽度',end = ' ')
    print(distance_lateral)
    print('胫骨外侧棘高度',end = ' ')
    print(height_tip)
    print(distance_lateral/height_tip)
else:
    distance_lateral = distance(up_feature_right, calculate_feet(up_feature_right, DLL,DLR))
    height_tip = distance(down_tips[1], calculate_feet(down_tips[1], DLL,DLR))
    print('外侧膝关节间缝隙宽度',end = ' ')
    print(distance_lateral)
    print('胫骨外侧棘高度',end = ' ')
    print(height_tip)
    print(distance_lateral/height_tip)


# In[86]:


# Hough检测直线、测量（弃用）
imgcp = img.copy()
rho = 1  # distance resolution in pixels of the Hough grid
theta = np.pi / 180  # angular resolution in radians of the Hough grid
threshold = 60  # minimum number of votes (intersections in Hough grid cell)
min_line_length = 10  # minimum number of pixels making up a line
max_line_gap = 20  # maximum gap in pixels between connectable line segments

# Run Hough on edge detected image
# Output "lines" is an array containing endpoints of detected line segments
lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                    min_line_length, max_line_gap)
    
for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(imgcp,(x1,y1),(x2,y2),(255,0,0),1)


# In[55]:


dirct = {1:2}
print(dirct)
del dirct[1]
print(dirct)


# In[57]:


for i in range(0,0):
    print(i)


# In[40]:


a = [1,2,3,4]
print(1 not in a)

