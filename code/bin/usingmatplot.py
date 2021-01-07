import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
# fig = plt.figure()
# fig.suptitle("no axes on this figure")
#
# fig,ax_1st=plt.subplots(2,2)
#
# a = pandas.DataFrame(np.random.rand(4,5), columns= list("abcde"))
# a_asndarray = a.values
#
# b= np.matrix([[1,2],[3,4]])
# b_asarray = np.asarray(b)
#
# x = np.linspace(0,2,100)
# plt.plot(x,x,label="linear")
# plt.plot(x,x**2,label="quadratic")
# plt.plot(x,x**3,label="cubic")
#
# plt.xlabel("x label")
# plt.ylabel("y label")
#
# plt.title("simple plot")
# plt.legend()
# plt.show()


column_header = ["id","date","all_squat_count","success_count","errors","hip_error","knee_error","back_error"]

dates = pd.date_range("20190101", periods=30)

all_squat_count_id1 = np.random.random_integers(40,60,size=(30,))

percentage = 0.5
success_id1 = []
for i in all_squat_count_id1:
    success_id1.append(i*percentage)
    percentage = min(percentage+0.02,1)

errors = all_squat_count_id1-success_id1

df = pd.DataFrame(columns = column_header)

df["date"] = dates

df["all_squat_count"] = all_squat_count_id1


df.iloc[:,3]= success_id1

df.iloc[:,4]= errors

df["id"] = 1


date_index = range(len(df["date"]))

#########
plt.figure(1)                # the first figure
ax1 = plt.subplot(211)             # the first subplot in the first figure
ax1.bar(date_index,df["all_squat_count"],align="center",color="darkblue")
ax1.xaxis.set_ticks_position("bottom")

ax1.yaxis.set_ticks_position("left")

plt.xticks(date_index,rotation=0,fontsize="small")

plt.ylabel("Total Squat Number")

plt.title("User 1 Squat History-January")
############


ax2 = plt.subplot(212)             # the second subplot in the first figure
ax2.bar(date_index,df["errors"],align="center",color="blue")
ax2.yaxis.set_ticks_position("left")

plt.xticks(date_index,rotation=0,fontsize="small")
plt.xlabel("Dates")
plt.ylabel("Error")

plt.savefig("squat_plot_6.png",dpi=400,bbox_inches="tight")

