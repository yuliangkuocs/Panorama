# Panorama - Multiple Image Stitching

## 環境
- openCV 3
- python 3
- numpy

## 流程
- 主程式為Panorama.py，其中的shrink_times代表一開始data要經過幾次gaussian pyramid down處理
- 將要stitch的圖片放到'images'資料夾，stitch順序要用檔名來表示，圖片格式皆為'DJI__{num}.JPG'
- 移到'Panorama'資料夾，輸入command: python3 code/Panorama.py 即可開始程式

## 待做
- 此程式採用blending的作法為 alpha blending，此方式速度較慢且多張照片的重複區域會較模糊，理論上multi-band blending效果會較好且更快，但自己實作失敗，尚待研究
- 有些許data算出來的homography matrix會有不好的結果，造成warp image失常，需要做例外處理
- 拼接出來的照片重疊邊界會有點黑邊瑕疵，可能是alpha blending的問題，尚須優化
