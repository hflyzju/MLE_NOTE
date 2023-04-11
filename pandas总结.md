

### 1. creat read save

```python
# https://www.kaggle.com/code/hxshine/exercise-creating-reading-and-writing/edit
# columns name
fruits = pd.DataFrame({"Apples":[30], "Bananas":[21]})
# with index
fruits = pd.DataFrame({"Apples":[35, 41], "Bananas":[21, 34]}, index=['2017 Sales', '2018 Sales'])
# Series: only one column and without columns name
ingredients = pd.Series(["4 cups","1 cup","2 large","1 can"], index=["Flour", "Milk", "Eggs", "Spam"], name="Dinner")

```

### 2. Exercise: Indexing, Selecting & Assigning

```python
# https://www.kaggle.com/code/hxshine/exercise-indexing-selecting-assigning/edit
desc = reviews.loc[:, "description"]
first_description = reviews.loc[0, "description"]
first_row = reviews.iloc[0]
first_descriptions = reviews.loc[:9, "description"]
sample_reviews = reviews.iloc[[1,2,3,5,8]]
df = reviews.loc[[0,1,10,100], ["country", "province", "region_1", "region_2"]]
df = reviews.loc[:99, ["country", "variety"]]
italian_wines = reviews.loc[(reviews.country == 'Italy')]
top_oceania_wines = reviews.loc[((reviews.country == 'Australia') | (reviews.country == 'New Zealand')) & (reviews.points >= 95)]

```

### 3. Summary Functions and Maps

```python
median_points = reviews.points.median()
countries = reviews.country.unique()
reviews_per_country = reviews.country.value_counts()
centered_price = reviews.price - reviews.price.mean()
centered_price = reviews['price'].apply(lambda x: x - mean_price)

bargain_idx = (reviews.points / reviews.price).idxmax()
bargain_wine = reviews.loc[bargain_idx, 'title']

n_trop = reviews.description.apply(lambda x: 1 if 'tropical' in x else 0).sum()
n_fruity = reviews.description.apply(lambda x: 1 if 'fruity' in x else 0).sum()
descriptor_counts = pd.Series([n_trop, n_fruity], index=['tropical', 'fruity'])


def get_score(x):
#     print(x)
    if x.country == "Canada":
        return 3
    else:
        if x.points >= 95:
            return 3
        elif x.points >= 85:
            return 2
        else:
            return 1
star_ratings = reviews.apply(lambda x: get_score(x), axis=1)


def stars(row):
    if row.country == 'Canada':
        return 3
    elif row.points >= 95:
        return 3
    elif row.points >= 85:
        return 2
    else:
        return 1

star_ratings = reviews.apply(stars, axis='columns')

```


### 4. Exercise: Grouping and Sorting

```python


```