# Research Software Lifecycle Analysis - Thesis Project

Please make sure install all packages in requirements.txt before you run all scripts.

All scripts are running in Python.


## Step 1 : Get research software list from RSD
In this step you need to have an API from RSD(Research Software Directory) and put it into .env file under the root directory.

Log in into Resaech Software Repository, You can find the Option *API Access Tokens* in the settings.

``RSD_TOKEN= "Your Token here"``

You will get the research software list and also named followed by the time.

*01_rsd_sofware_all_timestamp.csv*

There will be many *repo_urls* is not listed on GitHub, in this Research we are going to focus on GitHub repositories.

## Step 2: Get GitHub stats follwing the list

Please make sure you all put your GitHub TOKEN into .env

You can create your GitHub token in the developer settings.

``GITHUB_TOKEN= "Your token here"``

In this step, You will get four files including all GitHub activities info from all research softwares if they have accessable GitHub repositories.

*02_gh_commit_weekly_timestamp*

*02_gh_issues_timestamp*

*02_gh_releases_timestamp*

*02_gh_repo_meta_timestamp*

