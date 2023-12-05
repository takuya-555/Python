import cv2

# カスケードファイルの読み込み
cascade_path = "haarcascade_frontalface_default.xml"
cascade = cv2.CascadeClassifier(cascade_path)

# 画像の読み込み
img_path = "sample.png"
img = cv2.imread(img_path)

# グレースケールに変換
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 顔の検出
faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# 顔の周りに矩形を描画
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# 結果を表示
cv2.imshow("result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()