import copy
import time
import pandas as pd

from GA import GAScheduler

# read input file
pt_tmp = pd.read_excel("data\JSP_dataset.xlsx",sheet_name="Processing Time",index_col =[0])
ms_tmp = pd.read_excel("data\JSP_dataset.xlsx",sheet_name="Machines Sequence",index_col =[0])

num_job = pt_tmp.shape[0]
num_mc = pt_tmp.shape[1]
pt = [list(map(int, pt_tmp.iloc[i])) for i in range(num_job)]
ms = [list(map(int,ms_tmp.iloc[i])) for i in range(num_job)]
#print(pt)
#print(ms)

# every iteration is a simulation for specified parameters
while True:
	temp_pt = copy.deepcopy(pt)
	temp_ms = copy.deepcopy(ms)
	# current data
	print("Data :")
	print('\tthe number of jobs : ', num_job)
	print('\tthe number of machines :', num_mc)
	print("\n")
	choice = input("Continue with the data above? [y/n]: ")
	if choice == "y":
		population_size = int(input("population : "))
		num_iteration = int(input("generation(the number of iteration) : "))
		start_time = time.time()

		# GA
		s = GAScheduler(temp_pt, temp_ms)
		sequence_best, Tbest = s.run_genetic(population_size=population_size, num_iteration=num_iteration, verbose=True)
		stop_time = time.time()

		print("optimal sequence",sequence_best)
		print("optimal value:%f"%Tbest)
		print("The calculation takes " + str(stop_time - start_time) + " seconds")

		# draw Gantt chart
		draw = input("draw the Gantt chart of the best strategy ? [y/n] ")
		if draw == "n" or draw == "N":
			continue
		else:
			print("drawing...")
			s.draw_Gnatt(sequence_best)
		del s
	elif choice == "n":
		break
	else:
		print("Please enter again.")
	del temp_pt, temp_ms
