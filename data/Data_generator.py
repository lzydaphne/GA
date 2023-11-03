import xlsxwriter
import numpy as np

num_job = int(input("The number of jobs : "))
num_family = int(input("The number of families : "))
num_job_in_family = int(num_job/num_family)
setup_time_min = 2
setup_time_max = 8
Processing_time_min = 2
Processing_time_max = 24

workbook = xlsxwriter.Workbook('dataset_100.xlsx')
Job_type = workbook.add_worksheet("Job Type")

Job_type.write(0, 1, 'type')
Job_type.write(0, 2, 'setup')
for i in range(0, num_family):
    family_setup_time = np.random.randint(setup_time_min, setup_time_max)
    for j in range(1, num_job_in_family+1):
        Job_type.write(i*num_job_in_family+j, 0, 'J'+str(i*num_job_in_family+j))
        Job_type.write(i*num_job_in_family+j, 1, str(i+1))
        Job_type.write(i*num_job_in_family+j, 2, family_setup_time)

Processing_time = workbook.add_worksheet("Processing Time")
Processing_time.write(0, 1, 'time')
for i in range(1, num_job+1):
    random_processing_time = np.random.randint(Processing_time_min, Processing_time_max)
    Processing_time.write(i, 0, 'J'+str(i))
    Processing_time.write(i, 1, random_processing_time)

workbook.close()