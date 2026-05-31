{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "dbf62c74-904c-4be9-a9c8-c5a3d4fb7148",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import string\n",
    "import re\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.naive_bayes import MultinomialNB\n",
    "from sklearn.metrics import accuracy_score, classification_report, confusion_matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "df7ff01f-5c60-4ae3-8fa8-de6260787b4c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>category</th>\n",
       "      <th>rating</th>\n",
       "      <th>label</th>\n",
       "      <th>text_</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Home_and_Kitchen_5</td>\n",
       "      <td>5.0</td>\n",
       "      <td>CG</td>\n",
       "      <td>Love this!  Well made, sturdy, and very comfor...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Home_and_Kitchen_5</td>\n",
       "      <td>5.0</td>\n",
       "      <td>CG</td>\n",
       "      <td>love it, a great upgrade from the original.  I...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Home_and_Kitchen_5</td>\n",
       "      <td>5.0</td>\n",
       "      <td>CG</td>\n",
       "      <td>This pillow saved my back. I love the look and...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Home_and_Kitchen_5</td>\n",
       "      <td>1.0</td>\n",
       "      <td>CG</td>\n",
       "      <td>Missing information on how to use it, but it i...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Home_and_Kitchen_5</td>\n",
       "      <td>5.0</td>\n",
       "      <td>CG</td>\n",
       "      <td>Very nice set. Good quality. We have had the s...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "             category  rating label  \\\n",
       "0  Home_and_Kitchen_5     5.0    CG   \n",
       "1  Home_and_Kitchen_5     5.0    CG   \n",
       "2  Home_and_Kitchen_5     5.0    CG   \n",
       "3  Home_and_Kitchen_5     1.0    CG   \n",
       "4  Home_and_Kitchen_5     5.0    CG   \n",
       "\n",
       "                                               text_  \n",
       "0  Love this!  Well made, sturdy, and very comfor...  \n",
       "1  love it, a great upgrade from the original.  I...  \n",
       "2  This pillow saved my back. I love the look and...  \n",
       "3  Missing information on how to use it, but it i...  \n",
       "4  Very nice set. Good quality. We have had the s...  "
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df = pd.read_csv(\"C:/Users/bohar/Desktop/fake_review (1)/fake reviews dataset.csv\")\n",
    "\n",
    "df.head()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "b5822057-7315-49d9-a325-a87e42378a88",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Before rename: Index(['category', 'rating', 'label', 'text_'], dtype='object')\n",
      "After rename: Index(['category', 'rating', 'label', 'review'], dtype='object')\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>review</th>\n",
       "      <th>label</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Love this!  Well made, sturdy, and very comfor...</td>\n",
       "      <td>genuine</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>love it, a great upgrade from the original.  I...</td>\n",
       "      <td>genuine</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>This pillow saved my back. I love the look and...</td>\n",
       "      <td>genuine</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Missing information on how to use it, but it i...</td>\n",
       "      <td>genuine</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Very nice set. Good quality. We have had the s...</td>\n",
       "      <td>genuine</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                              review    label\n",
       "0  Love this!  Well made, sturdy, and very comfor...  genuine\n",
       "1  love it, a great upgrade from the original.  I...  genuine\n",
       "2  This pillow saved my back. I love the look and...  genuine\n",
       "3  Missing information on how to use it, but it i...  genuine\n",
       "4  Very nice set. Good quality. We have had the s...  genuine"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# check original columns\n",
    "print(\"Before rename:\", df.columns)\n",
    "\n",
    "# remove extra spaces (VERY IMPORTANT)\n",
    "df.columns = df.columns.str.strip()\n",
    "\n",
    "# rename safely\n",
    "if 'text_' in df.columns:\n",
    "    df = df.rename(columns={\"text_\": \"review\"})\n",
    "elif 'text' in df.columns:\n",
    "    df = df.rename(columns={\"text\": \"review\"})\n",
    "\n",
    "# check again\n",
    "print(\"After rename:\", df.columns)\n",
    "\n",
    "# convert label\n",
    "df['label'] = df['label'].apply(lambda x: \"genuine\" if x == \"CG\" else \"fake\")\n",
    "\n",
    "# keep only needed columns\n",
    "df = df[['review', 'label']]\n",
    "\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "60a5c5a6-9ef3-42a1-9955-bedf56db97d8",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>review</th>\n",
       "      <th>label</th>\n",
       "      <th>length</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>love this well made sturdy and very comfortabl...</td>\n",
       "      <td>genuine</td>\n",
       "      <td>69</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>love it a great upgrade from the original i ve...</td>\n",
       "      <td>genuine</td>\n",
       "      <td>77</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>this pillow saved my back i love the look and ...</td>\n",
       "      <td>genuine</td>\n",
       "      <td>66</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>missing information on how to use it but it is...</td>\n",
       "      <td>genuine</td>\n",
       "      <td>78</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>very nice set good quality we have had the set...</td>\n",
       "      <td>genuine</td>\n",
       "      <td>83</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                              review    label  length\n",
       "0  love this well made sturdy and very comfortabl...  genuine      69\n",
       "1  love it a great upgrade from the original i ve...  genuine      77\n",
       "2  this pillow saved my back i love the look and ...  genuine      66\n",
       "3  missing information on how to use it but it is...  genuine      78\n",
       "4  very nice set good quality we have had the set...  genuine      83"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import re\n",
    "\n",
    "def clean_text(text):\n",
    "    text = text.lower()\n",
    "    text = re.sub(r'\\W', ' ', text)\n",
    "    text = re.sub(r'\\s+', ' ', text)\n",
    "    return text\n",
    "\n",
    "df['review'] = df['review'].apply(clean_text)\n",
    "\n",
    "# extra feature (optional but good for viva)\n",
    "df['length'] = df['review'].apply(len)\n",
    "\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "d3cecf34-8e22-4357-a2b3-cdcaa1d92766",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "X = df['review']\n",
    "y = df['label']\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(\n",
    "    X, y, test_size=0.2, random_state=42\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "f8ec40d2-8adf-48e7-b176-65a647082f8c",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "\n",
    "tfidf = TfidfVectorizer(stop_words='english', max_features=5000)\n",
    "\n",
    "X_train_tfidf = tfidf.fit_transform(X_train)\n",
    "X_test_tfidf = tfidf.transform(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "dd91167a-df57-4ea3-8bfb-5d1c3fbce22d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Logistic Regression Accuracy: 0.8601459131940151\n",
      "\n",
      "Report:\n",
      "               precision    recall  f1-score   support\n",
      "\n",
      "        fake       0.85      0.87      0.86      4071\n",
      "     genuine       0.87      0.85      0.86      4016\n",
      "\n",
      "    accuracy                           0.86      8087\n",
      "   macro avg       0.86      0.86      0.86      8087\n",
      "weighted avg       0.86      0.86      0.86      8087\n",
      "\n"
     ]
    }
   ],
   "source": [
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.metrics import accuracy_score, classification_report\n",
    "\n",
    "lr_model = LogisticRegression(max_iter=200)\n",
    "lr_model.fit(X_train_tfidf, y_train)\n",
    "\n",
    "lr_pred = lr_model.predict(X_test_tfidf)\n",
    "\n",
    "print(\"Logistic Regression Accuracy:\", accuracy_score(y_test, lr_pred))\n",
    "print(\"\\nReport:\\n\", classification_report(y_test, lr_pred, zero_division=0))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "a1e6aee6-3e16-44e2-8d73-ff781532caab",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Naive Bayes Accuracy: 0.8436997650550266\n"
     ]
    }
   ],
   "source": [
    "from sklearn.naive_bayes import MultinomialNB\n",
    "\n",
    "nb_model = MultinomialNB()\n",
    "nb_model.fit(X_train_tfidf, y_train)\n",
    "\n",
    "nb_pred = nb_model.predict(X_test_tfidf)\n",
    "\n",
    "print(\"Naive Bayes Accuracy:\", accuracy_score(y_test, nb_pred))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "c3ceba2a-591b-4da1-9208-89f8e9d48211",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAjcAAAHHCAYAAABDUnkqAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjgsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvwVt1zgAAAAlwSFlzAAAPYQAAD2EBqD+naQAAMxVJREFUeJzt3QmcjXX///GPbWzZdxKpbAlFlrTcIlMpUQp1RxKVRHT7RQtFIbJUlAh5VOJHdOuuFErdFSlbG0qIFlsLooy4/o/39/8453fOzBlmmJkz853X8/G4mHOd67rO91wz13Xe57tcV54gCAIDAADwRN54FwAAACAjEW4AAIBXCDcAAMArhBsAAOAVwg0AAPAK4QYAAHiFcAMAALxCuAEAAF4h3AAAAK8QbgCkKk+ePPbwww+ne72tW7e6dV944YVMKZdvqlevbrfccku8iwF4g3ADZHMKCAoKmj788MMUz+sOKlWrVnXPX3XVVZYT7dy50/71r39Z7dq1rUiRIla0aFFr1KiRPfroo/b777/Hu3gAcpj88S4AgLQpVKiQzZo1yy688MKo+e+//7798MMPVrBgQcuJPv30U7vyyivtjz/+sH/+858u1Mhnn31mo0aNsg8++MDeeecd89nGjRstb16+awIZhXAD5BAKAHPnzrWnnnrK8uf/v0NXgUeBYM+ePZbTqFamQ4cOli9fPluzZo2ruYn02GOP2dSpU81HqnH766+/rHDhwjk2mALZFV8VgByiS5cu9ssvv9jixYvD85KSkmzevHl24403xlznwIEDdu+997pmK32A1qpVy5544gn3wRrp0KFD1r9/fytXrpwVK1bM2rVr52qDYvnxxx/t1ltvtQoVKrhtnn322TZ9+vQTek/PPfec2964ceNSBBvRazz44INR85555hn3mnrtypUr21133ZWi6eof//iH1atXzz7//HO75JJLXFPXmWee6fZVqLaradOmLlhonyxZsiRqffUzUjPfhg0b7IYbbrDixYtbmTJlrF+/fi6QRJoxY4ZdeumlVr58eVemunXr2rPPPhuzX42aDd9++21r3Lixe229/1h9bg4fPmyPPPKInXXWWa7GTq+tGrvI3728++67dtFFF7lmvJIlS9o111xj69evj/leNm3a5F5Dy5UoUcK6d+9uBw8ePO7vCMiJCDdADqEPwObNm9srr7wSnvfWW2/Z3r17rXPnzimWV4BRSBk/frxdfvnlLkDog3zgwIE2YMCAqGVvu+02mzBhgrVp08Y1BRUoUMDatm0bs29Ms2bNXBjo06ePPfnkky409OjRw62fXgsXLnQf8h07dkzT8vqgVphRqBk7dqxdd911LiCo3AoEkX777TcXJhRiRo8e7YKH9tOcOXPc/6oJ03tVANTr79+/P8XrKdgozIwcOdItr1qzXr16RS2jIFOtWjW7//77XZkUJHv37m2TJk2K2fykkHrZZZe5fdewYcNU36fCTcuWLW3ixIn2wAMP2GmnnWarV68OL6PfQWJiou3atcstr9/pxx9/bC1atHAdumO9F71HvRf9rL5ceg3ASwGAbG3GjBmqZgk+/fTTYOLEiUGxYsWCgwcPuueuv/76oGXLlu7natWqBW3btg2v99prr7n1Hn300ajtdezYMciTJ0+wadMm93jt2rVuud69e0ctd+ONN7r5Q4cODc/r0aNHUKlSpWDPnj1Ry3bu3DkoUaJEuFxbtmxx66rsx1KqVKmgQYMGadoPu3btChISEoI2bdoER44cCc/XPtFrTZ8+PTzvkksucfNmzZoVnrdhwwY3L2/evMGKFSvC899+++0UZdV71rx27dpFlUH7SPPXrVsXnhd6z5ESExODGjVqRM3T70frLlq0KMXyeq5bt27hx9onkb/LWBo2bBiUL18++OWXX8LzVC69v65du6Z4L7feemvU+h06dAjKlClzzNcAcipqboAcRN+4//zzT/vPf/7jvoXr/9SapN58803Xl6Vv375R89VMpVod1fqElpPky91zzz1Rj7XOq6++aldffbX7WX18QpNqEFSDFFmzkBb79u1zzWBpoZoKNcOpXJGdb3v27Omajd54442o5U855ZSoGi3VWqlJpk6dOq42JyT08+bNm1O8pmqJIt19991R+0xU8xSifaD9oaYwbU+PI51++uluXx2PyvnVV1/Zt99+G/P5n3/+2dauXeuamUqXLh2eX79+fVcrFFm+kDvuuCPqsZqz1Myp3wHgG8INkIOoT0zr1q1dJ+L58+fbkSNHUm3S+f77713zTfLwoA/30POh/xUWzjjjjKjlFAYi7d692/VtmTJliitH5KT+G6ImkvRQKInVHJTa+4lVroSEBKtRo0b4+ZBTTz3V9TWJpL4majZKPi/UjJWc+rxE0j7Svops9vnoo4/c7yTU70X7Q01UEivcpMWwYcPcvq5Zs6adc845rilR/YeOty9Cv18FLDW3RVKzVqRSpUql+r6BnI7RUkAOo5oa1Vbs2LHDrrjiCveBmhWOHj3q/tdw7W7dusVcRjUH6aFOxKqBUI2MQkpGUq1VeuYn72QdS/Kw9N1331mrVq3c+1CfJgUnvQ/VnKivU2ifxarlOZaLL77Ybfvf//63Gwb//PPPu+1NnjzZ9Y86ESfzvoGchpobIIfR0GnVHqxYsSLVJilRJ9effvopRc2IRgCFng/9rw9hfZgm7/waKTSSSrVFqqmINWnEUHqoiUvNbGruOp5QeZOXS8Foy5Yt4eczUvJmIY040r5S5255/fXX3UgzdYy+/fbbXadj7Ye0hphjUXOTasTUgXz79u0uOIauFp3avgj9fsuWLetqkoDcinAD5DDqS6IROvqgUzhIjT5oFUQ02iaSagBUA6FaHwn9r5FAkZKPftI3f41OUhD58ssvU7yemq3SS/1AKlWq5PoBffPNNymeVzOXrlIsCg2qFVE5I2sbpk2b5pp/Yo3uOlnJRzw9/fTTUfssVBsSWR6VRcPDT4b6wiT/nWtUmoKUaJ9ppNXMmTOjhsHr96KaHv3ugdyMZikgB0qtWSiSgo+GEmsYsfqINGjQwH3wqalDnXJDfWz0Ianhybp+jD6YL7jgAlu6dKmrpUhOQ6ffe+891wlXTWO6psuvv/7qOhKrw69+Tg/1+1iwYIH7MFY5Iq9QrG2q1kLD30M1R4MHD3bDlzW0XcPcVXOhcp9//vlu3YymGiG9jl5v+fLl9tJLL7naMu1L0RB0BS7ta9Xc6CrLuuigarDU6fdEab/qWj3aF6rB0dWadY0eDb8PGTNmjAtZ2j8aiq8aMIUv9SE6kfuBAV6J93AtAGkfCn4syYeCy/79+4P+/fsHlStXDgoUKBCcddZZwZgxY4KjR49GLffnn38Gffv2dUODixYtGlx99dXB9u3bUwwFl507dwZ33XVXULVqVbfNihUrBq1atQqmTJkSXiatQ8FDfvrpJ1fOmjVrBoUKFQqKFCkSNGrUKHjssceCvXv3Ri2rod+1a9d2r12hQoXgzjvvDH777beoZTQU/Oyzz07TPhKVVe8p+fDpr7/+2g2d1/B7DVvv06eP21eRFi5cGNSvX9+Vu3r16sHjjz/uhqVrfe2H4712rKHgGr7fpEmToGTJkkHhwoXd+9W+SEpKilpvyZIlQYsWLdwyxYsXd783lTlS6L3s3r075t9VZBkBX+TRP/EOWACQnYQuoqemNvVfAZCz0OcGAAB4hXADAAC8QrgBAABeoc8NAADwCjU3AADAK4QbAADglVx3ET9dOl2XpNdl5JPfJwYAAGRP6kWj28nohsC6Bc2x5Lpwo2CT/K7AAAAgZ9C91k499dRjLpPrwo1qbEI7p3jx4vEuDgAASIN9+/a5yonQ5/ix5LpwE2qKUrAh3AAAkLOkpUsJHYoBAIBXCDcAAMArhBsAAOAVwg0AAPAK4QYAAHiFcAMAALxCuAEAAF4h3AAAAK8QbgAAgFcINwAAwCuEGwAA4BXCDQAA8ArhBgAAeIVwAwAAvEK4AQAAXskf7wL4pvqgN+JdBCDb2jqqbbyLACAXoOYGAAB4hXADAAC8QrgBAABeIdwAAACvEG4AAIBXGC0FAOnEqEgge4+MpOYGAAB4hXADAAC8QrgBAABeIdwAAACvEG4AAIBXCDcAAMArhBsAAOAVwg0AAPAK4QYAAHiFcAMAALxCuAEAAF4h3AAAAK8QbgAAgFfiHm4mTZpk1atXt0KFClnTpk1t5cqVx1x+woQJVqtWLStcuLBVrVrV+vfvb3/99VeWlRcAAGRvcQ03c+bMsQEDBtjQoUNt9erV1qBBA0tMTLRdu3bFXH7WrFk2aNAgt/z69ett2rRpbhv3339/lpcdAABkT3ENN+PGjbOePXta9+7drW7dujZ58mQrUqSITZ8+PebyH3/8sbVo0cJuvPFGV9vTpk0b69Kly3FrewAAQO4Rt3CTlJRkq1atstatW/9fYfLmdY+XL18ec50LLrjArRMKM5s3b7Y333zTrrzyylRf59ChQ7Zv376oCQAA+Ct/vF54z549duTIEatQoULUfD3esGFDzHVUY6P1LrzwQguCwP7++2+74447jtksNXLkSHvkkUcyvPwAACB7inuH4vRYtmyZjRgxwp555hnXR2f+/Pn2xhtv2PDhw1NdZ/DgwbZ3797wtH379iwtMwAAyCU1N2XLlrV8+fLZzp07o+brccWKFWOu89BDD9nNN99st912m3t8zjnn2IEDB6xXr172wAMPuGat5AoWLOgmAACQO8St5iYhIcEaNWpkS5cuDc87evSoe9y8efOY6xw8eDBFgFFAEjVTAQAAxK3mRjQMvFu3bta4cWNr0qSJu4aNamI0ekq6du1qVapUcf1m5Oqrr3YjrM4991x3TZxNmza52hzND4UcAACQu8U13HTq1Ml2795tQ4YMsR07dljDhg1t0aJF4U7G27Zti6qpefDBBy1Pnjzu/x9//NHKlSvngs1jjz0Wx3cBAACykzxBLmvP0VDwEiVKuM7FxYsXz/DtVx/0RoZvE/DF1lFtzQcc50DWH+vp+fzOUaOlAAAAjodwAwAAvEK4AQAAXiHcAAAArxBuAACAVwg3AADAK4QbAADgFcINAADwCuEGAAB4hXADAAC8QrgBAABeIdwAAACvEG4AAIBXCDcAAMArhBsAAOAVwg0AAPAK4QYAAHiFcAMAALxCuAEAAF4h3AAAAK8QbgAAgFcINwAAwCuEGwAA4BXCDQAA8ArhBgAAeIVwAwAAvEK4AQAAXiHcAAAArxBuAACAVwg3AADAK4QbAADgFcINAADwCuEGAAB4hXADAAC8QrgBAABeIdwAAACvEG4AAIBXCDcAAMArhBsAAOAVwg0AAPAK4QYAAHiFcAMAALxCuAEAAF4h3AAAAK8QbgAAgFcINwAAwCuEGwAA4BXCDQAA8ArhBgAAeIVwAwAAvEK4AQAAXiHcAAAArxBuAACAVwg3AADAK4QbAADgFcINAADwCuEGAAB4hXADAAC8QrgBAABeIdwAAACvEG4AAIBXCDcAAMArhBsAAOAVwg0AAPAK4QYAAHiFcAMAALxCuAEAAF4h3AAAAK8QbgAAgFcINwAAwCuEGwAA4JW4h5tJkyZZ9erVrVChQta0aVNbuXLlMZf//fff7a677rJKlSpZwYIFrWbNmvbmm29mWXkBAED2lj+eLz5nzhwbMGCATZ482QWbCRMmWGJiom3cuNHKly+fYvmkpCS77LLL3HPz5s2zKlWq2Pfff28lS5aMS/kBAED2E9dwM27cOOvZs6d1797dPVbIeeONN2z69Ok2aNCgFMtr/q+//moff/yxFShQwM1TrQ8AAEDcm6VUC7Nq1Spr3br1/xUmb173ePny5THXWbhwoTVv3tw1S1WoUMHq1atnI0aMsCNHjqT6OocOHbJ9+/ZFTQAAwF9xCzd79uxxoUQhJZIe79ixI+Y6mzdvds1RWk/9bB566CEbO3asPfroo6m+zsiRI61EiRLhqWrVqhn+XgAAQPYR9w7F6XH06FHX32bKlCnWqFEj69Spkz3wwAOuOSs1gwcPtr1794an7du3Z2mZAQBALulzU7ZsWcuXL5/t3Lkzar4eV6xYMeY6GiGlvjZaL6ROnTqupkfNXAkJCSnW0YgqTQAAIHeIW82NgohqX5YuXRpVM6PH6lcTS4sWLWzTpk1uuZBvvvnGhZ5YwQYAAOQ+cW2W0jDwqVOn2syZM239+vV255132oEDB8Kjp7p27eqalUL0vEZL9evXz4UajaxSh2J1MAYAAIj7UHD1mdm9e7cNGTLENS01bNjQFi1aFO5kvG3bNjeCKkSdgd9++23r37+/1a9f313nRkHnvvvui+O7AAAA2Ulcw4306dPHTbEsW7YsxTw1Wa1YsSILSgYAAHKiHDVaCgAA4HgINwAAwCuEGwAA4BXCDQAA8ArhBgAAeIVwAwAAvEK4AQAAXiHcAAAArxBuAACAVwg3AADAK4QbAADgFcINAADwCuEGAAB4hXADAAC8QrgBAABeIdwAAACvEG4AAIBXCDcAAMArhBsAAOAVwg0AAPAK4QYAAHiFcAMAALxCuAEAAF4h3AAAAK8QbgAAQO4ON9WrV7dhw4bZtm3bMqdEAAAAWRlu7rnnHps/f77VqFHDLrvsMps9e7YdOnToZMoAAAAQ33Czdu1aW7lypdWpU8fuvvtuq1SpkvXp08dWr16dcSUDAADIyj435513nj311FP2008/2dChQ+3555+3888/3xo2bGjTp0+3IAhOdNMAAAAnLP+Jrnj48GFbsGCBzZgxwxYvXmzNmjWzHj162A8//GD333+/LVmyxGbNmnXiJQMAAMiKcKOmJwWaV155xfLmzWtdu3a18ePHW+3atcPLdOjQwdXiAAAAZPtwo9CijsTPPvustW/f3goUKJBimdNPP906d+6cUWUEAADIvHCzefNmq1at2jGXKVq0qKvdAQAAyPYdinft2mWffPJJivma99lnn2VUuQAAALIm3Nx11122ffv2FPN//PFH9xwAAECOCjdff/21Gwae3LnnnuueAwAAyFHhpmDBgrZz584U83/++WfLn/+ER5YDAADEJ9y0adPGBg8ebHv37g3P+/333921bTSKCgAAIJ7SXdXyxBNP2MUXX+xGTKkpSnQ7hgoVKtiLL76YGWUEAADIvHBTpUoV+/zzz+3ll1+2devWWeHCha179+7WpUuXmNe8AQAAyEon1ElG17Hp1atXxpcGAADgJJ1wD2CNjNq2bZslJSVFzW/Xrt3JlgkAACBrr1Cse0d98cUXlidPnvDdv/WzHDly5MRLAwAAkNWjpfr16+fuHaUrFRcpUsS++uor++CDD6xx48a2bNmyky0PAABA1tbcLF++3N59910rW7asuyu4pgsvvNBGjhxpffv2tTVr1pxciQAAALKy5kbNTsWKFXM/K+D89NNP7mcNDd+4cePJlAUAACDra27q1avnhoCraapp06Y2evRoS0hIsClTpliNGjVOvkQAAABZGW4efPBBO3DggPt52LBhdtVVV9lFF11kZcqUsTlz5pxMWQAAALI+3CQmJoZ/PvPMM23Dhg3266+/WqlSpcIjpgAAAHJEn5vDhw+7m2N++eWXUfNLly5NsAEAADkv3Oj2CqeddhrXsgEAAP6MlnrggQfcHcDVFAUAAJDj+9xMnDjRNm3aZJUrV3bDv3WfqUirV6/OyPIBAABkbrhp3759elcBAADIvuFm6NChmVMSAACAePS5AQAA8KrmRveSOtawb0ZSAQCAHBVuFixYkOLaN7pZ5syZM+2RRx7JyLIBAABkfri55pprUszr2LGjnX322e72Cz169Eh/KQAAALJbn5tmzZrZ0qVLM2pzAAAA8Qs3f/75pz311FNWpUqVjNgcAABA1jVLJb9BZhAEtn//fitSpIi99NJLJ14SAACAeISb8ePHR4UbjZ4qV66cNW3a1AUfAACAHBVubrnllswpCQAAQDz63MyYMcPmzp2bYr7maTg4AABAjgo3I0eOtLJly6aYX758eRsxYkRGlQsAACBrws22bdvs9NNPTzFfdwjXcwAAADkq3KiG5vPPP08xf926dVamTJmMKhcAAEDWhJsuXbpY37597b333nP3kdL07rvvWr9+/axz586ZU0oAAIDMGi01fPhw27p1q7Vq1cry5///qx89etS6du1KnxsAAJDzwk1CQoK7h9Sjjz5qa9eutcKFC9s555zj+twAAADk2NsvnHXWWXb99dfbVVddddLBZtKkSVa9enUrVKiQuxjgypUr07Te7Nmz3QUF27dvf1KvDwAAcnG4ue666+zxxx9PMX/06NEu7KSXaoEGDBhgQ4cOtdWrV1uDBg0sMTHRdu3adcz11DT2r3/9yy666KJ0vyYAAPBX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3Ov0prtOS48ePYJevXpFLfPf//43yJs3b/Dnn3+muAv72LFj3Z2c9VoAch5qbgDETbly5axt27b2wgsvuBoc/Rx57zLVuBw+fNjdxDSkQIEC7mau69evd4/1f9OmTaO227x586jH69atc6+hm6OGJtX+HD161LZs2ZKiXNdff73rC1SjRg3r2bOnu7Hq33//nQl7AEBm4K7gAOLeNBVqHpo0aVKm3eX59ttvj9lvJlbnZfXJ2bhxoy1ZssQWL15svXv3tjFjxtj777/vwhWA7I2aGwBxpb4wSUlJroZGtSmRzjjjDEtISLCPPvooPE/LqUNx3bp13eM6derYypUro9ZbsWJFis7LX3/9teuvk3zS9mNRx2P143nqqafcdXOWL19uX3zxRQa+cwCZhZobAHGVL1++cBOTfo5UtGhRu/POO23gwIFWunRpV8syevRoO3jwoPXo0cMto2vljB071i2jzsSrVq1yTVCR7rvvPmvWrJmrIdIy2q7CjmplJk6cmKJMWv/IkSOuuatIkSL20ksvubBTrVq1TN0XADIGNTcA4q548eJuimXUqFFulNPNN9/samA2bdpkb7/9tpUqVco9r8Cj0U0a0dSgQQObPHmyjRgxImob9evXd01K33zzjRsOrqHmQ4YMcUO8YylZsqRNnTrV9fXRumqeev31161MmTKZ8O4BZLQ86lWc4VsFAACIE2puAACAVwg3AADAK4QbAADgFcINAADwCuEGAAB4hXADAAC8QrgBAABeIdwAAACvEG4AAIBXCDcAAMArhBsAAOAVwg0AADCf/D9vkhOdZ5Oy4AAAAABJRU5ErkJggg==",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "from sklearn.metrics import accuracy_score\n",
    "\n",
    "models = [\"Logistic Regression\", \"Naive Bayes\"]\n",
    "\n",
    "acc_lr = accuracy_score(y_test, lr_pred)\n",
    "acc_nb = accuracy_score(y_test, nb_pred)\n",
    "\n",
    "accuracies = [acc_lr, acc_nb]\n",
    "\n",
    "plt.figure()\n",
    "plt.bar(models, accuracies)\n",
    "plt.title(\"Model Comparison\")\n",
    "plt.xlabel(\"Models\")\n",
    "plt.ylabel(\"Accuracy\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "c5770f3b-c227-40ff-85c2-9f35a001aad8",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 640x480 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "from sklearn.metrics import confusion_matrix\n",
    "import seaborn as sns\n",
    "\n",
    "cm = confusion_matrix(y_test, lr_pred)\n",
    "\n",
    "plt.figure()\n",
    "sns.heatmap(cm, annot=True, fmt='d')\n",
    "plt.title(\"Confusion Matrix\")\n",
    "plt.xlabel(\"Predicted\")\n",
    "plt.ylabel(\"Actual\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "d6e05bea-d1c8-4d03-8eaa-a9c5ee5fbe33",
   "metadata": {},
   "outputs": [
    {
     "name": "stdin",
     "output_type": "stream",
     "text": [
      "Enter review:  This is the best product ever!!! Everyone must buy this right now, totally life-changing!!!\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Prediction: fake\n"
     ]
    }
   ],
   "source": [
    "user_input = input(\"Enter review: \")\n",
    "\n",
    "clean = clean_text(user_input)\n",
    "vec = tfidf.transform([clean])\n",
    "\n",
    "print(\"Prediction:\", lr_model.predict(vec)[0])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "b60c705f-5e00-4624-8c67-17b358f0a91f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdin",
     "output_type": "stream",
     "text": [
      "Enter review:  good\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Prediction: genuine\n"
     ]
    }
   ],
   "source": [
    "import pickle\n",
    "\n",
    "model = pickle.load(open(\"model.pkl\", \"rb\"))\n",
    "vectorizer = pickle.load(open(\"vectorizer.pkl\", \"rb\"))\n",
    "\n",
    "text = input(\"Enter review: \")\n",
    "vec = vectorizer.transform([text])\n",
    "\n",
    "print(\"Prediction:\", model.predict(vec)[0])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "77839616-f3d2-494e-aee4-5845759bc44f",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "23e631dc-d346-498e-a73b-3e016f5d7fd4",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
