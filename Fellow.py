'''
#Poverty and Crime Rates Across U.S. States (2024)#

This project examines the association between state level poverty rates
and violent and property crime rates in the United States.

Data are drawn from 2024 ACS S1701 and OpenCrime state level crime data.

The analysis includes data cleaning and merging, descriptive statistics, visualization, 
and OLS regression with HC3 robust standard errors. 
A sensitivity analysis excludes Alaska to assess the influence of
an unusually high violent crime rate.

All analyses are observational and do not establish causality.
'''
from pathlib import Path
import pandas as pd
import json 

folder=Path(__file__).resolve().parent
census=pd.read_csv(folder/"S1701.csv")
with open(folder/"state-trends.json", "r", encoding="utf-8") as f:
	crime=json.load(f)

poverty=[]
for column in census.columns:
	if column.endswith("!!Percent below poverty level!!Estimate"):
		state=column.split("!!")[0]
		value=census.iloc[0][column]
		if pd.notna(value):
			value=str(value).replace("%","").strip()
			if value:
				rate=float(value)
				record={"State": state, "poverty": rate}
				poverty.append(record)

poverty_df=pd.DataFrame(poverty)
excluded= ["District of Columbia", "Puerto Rico"]
poverty_df=poverty_df[ ~poverty_df["State"].isin(excluded)].copy()

crime_records=[]
for state in crime:
	stateName=state["name"]
	for record in state["years"]:
		if record["year"]==2024:
			violent=record["violentRate"]
			property=record["propertyRate"]
			record={"State": stateName, "Violent": violent, "Property": property}
			crime_records.append(record)
crime_df=pd.DataFrame(crime_records)
crime_df=crime_df[~ crime_df["State"].isin(excluded)].copy()

merge_df=pd.merge(poverty_df,crime_df,on="State",how="inner")

assert len(merge_df)==50
assert merge_df["State"].is_unique
assert not merge_df[["poverty","Violent","Property"]].isna().any().any()

print(merge_df[["poverty","Violent","Property"]].describe())


import matplotlib.pyplot as plt

poverty=merge_df["poverty"]
violentCrime=merge_df["Violent"]
propertyCrime=merge_df["Property"]

plt.figure(figsize=(7,5))
plt.scatter(poverty,violentCrime,alpha=0.7)
plt.xlabel("Poverty Rate (%)")
plt.ylabel("Violent Crime Rate (per 100,000)")
plt.title("Poverty and Violent Crime Across U.S. States, 2024")
plt.tight_layout()
plt.savefig(folder / "figure1_violent.png", dpi=300)
plt.close()

plt.figure(figsize=(7,5))
plt.scatter(poverty,propertyCrime,alpha=0.7)
plt.xlabel("Poverty Rate (%)")
plt.ylabel("Property Crime Rate (per 100,000)")
plt.title("Poverty and Property Crime Across U.S. States, 2024")
plt.tight_layout()
plt.savefig(folder / "figure2_property.png", dpi=300)
plt.close()


import statsmodels.api as sm

def run_OLS(data, dependent, independent):
	X=sm.add_constant(data[independent])
	Y=data[dependent]
	model=sm.OLS(Y,X).fit(cov_type="HC3")
	return model

modelViolent=run_OLS(merge_df,"Violent","poverty")
modelProperty=run_OLS(merge_df,"Property","poverty")

ExAlaska=merge_df[merge_df["State"]!="Alaska"].copy()
violentRobust=run_OLS(ExAlaska,"Violent","poverty")
propertyRobust=run_OLS(ExAlaska,"Property","poverty")

def result(model, name):
	result={"Model":name,
			"Coef":model.params["poverty"],
			"SE":model.bse["poverty"],
			"p-value":model.pvalues["poverty"],
			"R-square":model.rsquared,
			"Nobs":int(model.nobs)
			}
	return result

results=[result(modelViolent,"ViolentFull"),
		result(modelProperty,"PropertyFull"),
		result(violentRobust,"Violent-ExAlaska"),
		result(propertyRobust,"Property-ExAlaska")
		]
result_df=pd.DataFrame(results)
print(result_df.round(3))

result_df.to_csv(folder/"regression_results.csv",index=False)
merge_df.to_csv(folder / "merged_2024_clean.csv", index=False)
