# Panorama - Multiple Image Stitching

### <font color='blue'>20190314 - 郭毓梁</font>

## Outline
- Enviroment
- Introduction
- Inplementation
- Result
- To Do

## Enviroment
- openCV 3
- python 3
- numpy

## Introduction
- 我的圖像拼接演算法架構如下：於下個段落詳細說明
    1. Data Loading
    2. Feature Detecting & Descripting
    3. Feature Matching
    4. Calculate Homography Matrix
    5. Stitch Image & Cut Padding
    6. Blending

## Inplementation
### Data Loading
- 首先需要Loading所有的照片，並且全部改成numpy的格式(ndarray)，也就是OpenCV在Python中處理影像的格式，並且將照片依照檔名依序排好，完成這些步驟才能開始一張一張拼接。
- 我會將每次拼接好的照片與下一張dataset中的照片拼接，每次都是跑StitchTwoImage(A, B)，A為拼接多張照片的大圖，B為原始dataset的下一張圖片

### Feature Detecting & Descripting
- 圖片拼接的關鍵就在於找出兩張圖中相同的特徵點，再去進一步算出兩個圖片的轉換矩陣，而特徵點要用Descriptor來描述，才能讓不同特徵點彼此做比較
- 傳統找特徵點有SIFT、SURF等演算法，而我使用ORB演算法，會比SIFT和SURF再快更多，但要注意的是不同特徵點的演算法其match的演算法也要跟著不同才會有好的效果
```python=
orb = cv2.ORB_create(MAX_FEATURES)
keypoints, descriptors = orb.detectAndCompute(image, None)
```

### Feature Matching
- 擁有了兩張圖各自許多個特徵點資料，現在要做的就是去比對兩張圖眾多特徵點中哪些才是相對應的特徵點，SIFT與SURF是去算他們不同特徵點中L2 Distance的error，越小則代表這兩個特徵點越像，而ORB則是算Norm Hamming的Distance，一樣越小代表越像，這些演算法OpenCV都有支援
- 一開始我並沒有注意到ORB應該使用Norm Hamming的Distance來做Matching，一樣使用L2 Distance來計算Error，結果轉換的Homography Matrix算出來並沒有很理想，後來改過來之後，homography matrix變好不少，讓拼接出來的圖片更有品質
```python=
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

# Sort match descriptors in the order of their distance.
matches = sorted(matches, key=lambda x: x.distance)

num_good_matches = int(len(matches) * GOOD_MATCH_PERCENT)
matches = matches[:num_good_matches]
```

### Calculate Homography Matrix
- 利用RANSAC演算法從眾多Match的點中找出inliers最多的homography matrix，此matrix即為兩圖的變換關係
- 偶爾算出來的matrix並不夠理想，可藉由回傳的status知道算出來的matrix所對應的inliers數目，便能藉此去掉不好的data，便繼續算下去
```python=
h, s = cv2.findHomography(points1, points2, cv2.RANSAC)

inliers_radio = float(np.sum(s)) / float(len(s))

isHomographyGood = inliers_radio >= 0.2
```


### Stitch Image & Cut Padding
- 接下來便是照著轉換矩陣來把兩張圖片拼接在一起，首先需要擴充Image的大小，才足以塞得下兩張照片
- 將圖片經過轉換矩陣得到新的圖片OpenCV也有支援，還能設定Border要用什麼形式，使用Reflect Border會讓之後blending有更好的效果，沒使用border的則是當作mask來使用
```python=
pad_raw = cv2.warpPerspective(raw_img, t, new_size, borderMode=cv2.BORDER_REFLECT)
raw_mask = cv2.warpPerspective(raw_img, t, new_size)

pad_warp = cv2.warpPerspective(stitch_img, t.dot(h), new_size, borderMode=cv2.BORDER_REFLECT)
warp_mask = cv2.warpPerspective(stitch_img, t.dot(h), new_size)
```

- 當圖片已經轉換好後，在進行拼接之前，我們先縮減圖片的大小，將不會用到的黑色padding區域給裁切掉，如此能讓之後blending的步驟更快完成，cut padding所用的mask則是兩張圖經過 'or' 運算後的結果，詳細cut padding的算法可在 <font color='red'>cutPadidng</font> function裡看
```python=
stitch_mask = np.logical_or(getMask(raw_mask), getMask(warp_mask))
stitch_mask = np.asarray(stitch_mask, dtype=np.uint8)

cut_pad_images = [pad_raw, pad_warp, raw_mask, warp_mask]
[cut_pad_raw, cut_pad_warp, cut_pad_raw_mask, cut_pad_warp_mask] = cutPadding(cut_pad_images, stitch_mask)
```

### Blending
- 最後就是將兩張圖經過blending拼捷在一起，常用的技術有兩種，一種是alpha blending，另一種事multi-band blending
- alpha blending可能會讓圖產生幽靈效果，且速度也比multi-band blending慢，單純就是兩張圖權重的相加
- multi-band blending更快且保有原圖片特性，但要選好mask區域否則會把圖片外白邊也混合進來

## Result
- 找feature和算homography matrix都非常快速，都能在1s內完成，主要慢的是stitch image以及blending的地方，尤其當圖越合越大，所需時間也會快速上升
- 目前速度狀況：（i5 CPU, 16G RAM, Mac Pro）
    - 合7張照片：157 s


- data 183-197
![](https://i.imgur.com/JFhkpE9.jpg)

- data 49-57
![](https://i.imgur.com/SVmXDmD.jpg)




