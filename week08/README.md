# 第 8 週作業：交叉驗證與網格搜尋

## 作業資訊

| 項目 | 說明 |
|------|------|
| 對應教科書 | Ch11 交叉驗證、Ch12 模型參數挑選和網格搜尋 |
| 繳交方式 | 在 Fork 的 week08/ 資料夾中建立三個檔案，發 PR 繳交 |
| 繳交期限 | 下週上課前 |
| PR 標題格式 | 學號_姓名_week08 |

---

## 第 1 題：5 折交叉驗證多模型比較（40 分）

### 任務說明

使用 5 折交叉驗證，正確比較至少四種分類模型在同一份資料集上的效能。

### 測試資料

請使用以下程式碼載入資料：

```python
from sklearn.datasets import load_breast_cancer
import pandas as pd

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

print(f"資料形狀：{X.shape}")
print(f"類別分布：\n{pd.Series(y).value_counts()}")
print(f"類別名稱：{data.target_names}")
```

### Python 程式要求

撰寫程式碼完成以下工作：

1. 載入 breast_cancer 資料集
2. 建立至少四種分類模型的 Pipeline（含 StandardScaler）：
   - LogisticRegression
   - KNeighborsClassifier
   - SVC
   - DecisionTreeClassifier
3. 對每個模型使用 cross_val_score 進行 5 折交叉驗證
4. 印出每個模型的 5 次分數、平均分數、標準差
5. 繪製長條圖或箱型圖比較各模型的交叉驗證結果
6. 找出最佳模型

### 作答內容

請建立 `week08/q1_cross_validation.txt`，依照以下格式填寫：

```
姓名：
學號：

=== 完整程式碼 ===
（貼上你撰寫的完整 Python 程式碼）

=== 交叉驗證結果 ===
LogisticRegression：5 次分數=???  平均=???  標準差=???
KNN：5 次分數=???  平均=???  標準差=???
SVM：5 次分數=???  平均=???  標準差=???
DecisionTree：5 次分數=???  平均=???  標準差=???

=== 最佳模型 ===
最佳模型：???
平均準確率：???
```

### 提示

```python
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

models = {
    'LogisticRegression': LogisticRegression(max_iter=5000),
    'KNN': KNeighborsClassifier(),
    'SVM': SVC(),
    'DecisionTree': DecisionTreeClassifier(random_state=42)
}

results = {}
for name, model in models.items():
    pipe = Pipeline([('scaler', StandardScaler()), ('clf', model)])
    scores = cross_val_score(pipe, X, y, cv=5, scoring='accuracy')
    results[name] = scores
    print(f"{name}：{scores}  平均={scores.mean():.4f}  標準差={scores.std():.4f}")
```

---

## 第 2 題：GridSearchCV 超參數調校（40 分）

### 任務說明

使用 GridSearchCV 對 SVM 進行系統化的超參數調校，找出最佳的 C 和 gamma 組合。

### Python 程式要求

使用第 1 題相同的 breast_cancer 資料集，完成以下工作：

1. 切割資料（test_size=0.2, random_state=42）
2. 建立 Pipeline：StandardScaler → SVC
3. 定義參數搜尋空間：
   - C：[0.1, 1, 10, 100]
   - gamma：['scale', 'auto', 0.01, 0.1]
   - kernel：['rbf', 'linear']
4. 使用 GridSearchCV（cv=5）進行搜尋
5. 印出最佳參數組合與對應的交叉驗證分數
6. 使用最佳模型對測試集預測，印出 Classification Report
7. 印出前 5 名的參數組合與分數

### 作答內容

請建立 `week08/q2_grid_search.txt`，依照以下格式填寫：

```
姓名：
學號：

=== 完整程式碼 ===
（貼上你撰寫的完整 Python 程式碼）

=== 搜尋空間大小 ===
總共測試了幾種組合：???

=== 最佳結果 ===
最佳參數：???
最佳交叉驗證分數：???
測試集準確率：???

=== 前 5 名參數組合 ===
1. ???
2. ???
3. ???
4. ???
5. ???

=== 最佳模型的 Classification Report ===
（貼上分類報告）
```

### 提示

```python
from sklearn.model_selection import GridSearchCV

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC())
])

param_grid = {
    'svm__C': [0.1, 1, 10, 100],
    'svm__gamma': ['scale', 'auto', 0.01, 0.1],
    'svm__kernel': ['rbf', 'linear']
}

grid = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)

print(f"最佳參數：{grid.best_params_}")
print(f"最佳交叉驗證分數：{grid.best_score_:.4f}")
print(f"測試集準確率：{grid.score(X_test, y_test):.4f}")

# 前 5 名
results_df = pd.DataFrame(grid.cv_results_)
top5 = results_df.nsmallest(5, 'rank_test_score')[['params', 'mean_test_score', 'rank_test_score']]
print(top5)
```

---

## 第 3 題：交叉驗證與網格搜尋觀念題（20 分）

### 作答內容

請建立 `week08/q3_concept.txt`，回答以下問題：

```
姓名：
學號：

Q1：為什麼交叉驗證比單次 train_test_split 更能反映模型的真實效能？
    如果只做一次 train_test_split，可能會遇到什麼問題？
A1：???

Q2：GridSearchCV 中的 cv=5 代表什麼意思？
    如果參數搜尋空間有 4 個 C 值 × 4 個 gamma 值 × 2 個 kernel，
    加上 cv=5，總共需要訓練幾次模型？請列出計算過程。
A2：???

Q3：SVM 的參數 C 控制什麼？C 值很大和很小分別會造成什麼效果？
    這跟過擬合/欠擬合有什麼關係？
A3：???
```

---

## 繳交 Checklist

- [ ] week08/q1_cross_validation.txt 包含完整程式碼與四模型交叉驗證比較
- [ ] week08/q2_grid_search.txt 包含完整程式碼與 GridSearchCV 結果
- [ ] week08/q3_concept.txt 包含三題觀念回答
- [ ] 已 push 到自己的 Fork
- [ ] 已發 PR，標題格式：學號_姓名_week08

## 常見問題

**Q：cross_val_score 和 GridSearchCV 都有 cv 參數，有什麼差別？**
cross_val_score 是用來評估單一模型的效能，GridSearchCV 則是在多組參數中搜尋最佳組合。兩者都用 K-Fold 交叉驗證，但目的不同。

**Q：GridSearchCV 跑很久怎麼辦？**
加上 n_jobs=-1 可以使用所有 CPU 核心平行運算。也可以先縮小搜尋空間測試程式是否正確，確認後再擴大搜尋。

**Q：Pipeline 裡的參數名稱 'svm__C' 是怎麼來的？**
格式是「步驟名稱__參數名稱」。Pipeline 中 SVC 的步驟名稱是 'svm'，所以它的 C 參數就寫成 'svm__C'。
