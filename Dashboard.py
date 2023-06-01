import influxdb_client, os, time
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
import feed_foward as fd

token = os.environ.get("INFLUXDB_TOKEN")
org = "vasco.conceicao3030@gmail.com"
url = "https://europe-west1-1.gcp.cloud2.influxdata.com"
bucket = "Test"
token = "Mmu4hHTXxJMsRmM-qGcdkVN6co65IeZeTT1hWAXr1YSiRoglzYy5xH-x0TdlUGHdMHTQih9PBG5hGl6Q6a-c_A=="

client = influxdb_client.InfluxDBClient(
    url=url,
    token=token,
    org=org
)

def standardise(item,max,min):
    
    return 0.8*((item - min)/(max - min))+0.1


def destandardise(item,max,min):
    
    return ((item-0.1)/0.8)*(max-min)+min
 
weights_matrix_between_input_hidden = [[1.179254531427120689e+01, 1.264563091847170995e+01], [-1.083832000213494773e+01, -9.791435223970211155e+00], [-1.581817680419104732e+01, -1.068864726790247843e+01], [1.498300209409340589e+01, 5.759502391600410220e+00]]
weights_matrix_between_hidden_output = [-7.342481064839878, -4.193543823694243, -10.481743416889536, -7.593006036882884]
bias_hidden_nodes = [[-1.189752817304256993e+01], [8.353218827017046877e+00], [1.325546480003319694e+01], [-1.024225307434639554e+01]]
bias_outputs = [14.536338707860855]

max_temp = 30.5
min_temp = 11
max_hum = 84.7725
min_hum = 40.9175
max_output = 12.2
min_output = 7.19999999999999

write_api = client.write_api(write_options=SYNCHRONOUS)


file = open(r"C:\Users\VascoConceicao\Desktop\Gsat.txt","r")
file_ = open(r"C:\Users\VascoConceicao\Desktop\Resultados.txt","r")

while(1):

    string = file.readline()
    string_ = file_.readline()
    #print(len(string))

    if len(string)<45:

        string_split = string.split(";")

        temperatura = float(string_split[1])
        humidade = float(string_split[4])

        inputs_array = [[standardise(temperatura,max_temp,min_temp)],[standardise(humidade,max_hum,min_hum)]]

        returned = fd.feed_forward(weights_matrix_between_input_hidden,weights_matrix_between_hidden_output,bias_hidden_nodes,bias_outputs,inputs_array)

        output = destandardise(returned[0],max_output,min_output)

        p = influxdb_client.Point("Table").field("Temperatura_BME", float(string_split[1])).field("Pressao_BME",float(string_split[2])).field("Altitude_BME", float(string_split[3])).field("Humidade_BME", float(string_split[4])) .field("Temperatura_Thermistor",float(string_split[5])).field("Iindice de transmissibilidade", float(string_))  
        write_api.write(bucket=bucket, org=org, record=p)

    else:

        string_split = string.split(";")
        
        temperatura = float(string_split[5])
        humidade = float(string_split[8])

        inputs_array = [[standardise(temperatura,max_temp,min_temp)],[standardise(humidade,max_hum,min_hum)]]

        returned = fd.feed_forward(weights_matrix_between_input_hidden,weights_matrix_between_hidden_output,bias_hidden_nodes,bias_outputs,inputs_array)

        output = destandardise(returned[0],max_output,min_output)

        p = influxdb_client.Point("Table").field("Latitude", float(string_split[3])).field("Longitude", float(string_split[4])).field("Altitude_GPS", float(string_split[5])).field("Temperatura_BME", float(string_split[6])).field("Pressure_BME",float(string_split[7])).field("Altitude_BME", float(string_split[8])).field("Humidity_BME", float(string_split[9])) .field("Temperature_Thermistor",float(string_split[10])).field("Transmission Rate", float(string_)) 
        write_api.write(bucket=bucket, org=org, record=p)

    time.sleep(1)

file.close()