# SmartCareer
Here is a repository for SmartCareer, based on DSSM, here is the structure
```text
├───
|   api.py           -- script to start app
│   Constants.py     -- Constant variable
│   job_data.py      -- Prepare data for model
│   main.py          -- Main script to run
│   model.py         -- Model structure
│   rating_preprocess.ipynb  -- Negative sampling process
│   README.md  
│   Recommender_system.ipynb   
│   SkillNer.ipynb   -- User spicy to do Ner
│   Talent Exchange.ipynb  -- Process of building knowledge graph
│
├───.ipynb_checkpoints
│
├───checkpoint  -- Folder which contains model weights for each echo
|
├───data
│       Simbert_None_value.npy   -- data matrix of "None" in minibert
├───static
│   ├───css
│   │       style.css
│   │
│   └───images
│           1200px-Flask_logo.svg.png
|
├───templates
│       home.html

```
<br><br>

## Data Sources
* [for project data](https://pwc-my.sharepoint.com/personal/wenguang_lin_pwc_com/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fwenguang%5Flin%5Fpwc%5Fcom%2FDocuments%2FSmartCareer&OR=Teams%2DHL&CT=1673247897780&clickparams=eyJBcHBOYW1lIjoiVGVhbXMtRGVza3RvcCIsIkFwcFZlcnNpb24iOiIyNy8yMzAxMDUwNTYwMCIsIkhhc0ZlZGVyYXRlZFVzZXIiOmZhbHNlfQ%3D%3D)
* [for bert weight](https://drive.google.com/drive/folders/1jEGAr7o_tukeOVQ55-sFnRnCeRS1zY4q) 


<br><br>

## How to run this model  

Check requirements.txt for all the necessary python packages

### Method 1

Run Recommender_system.ipynb for step-by-step understanding

### Method 2
- Change Constants.py, which contains the path of all data
- Run main.py

<br><br>

## App 
- run the following scripts in  
```python3
python api.py
```

## Remains to be done
- 求数据库里头的所有User和Job的embedding, 然后做外推计算
- Cold start

<br><br>