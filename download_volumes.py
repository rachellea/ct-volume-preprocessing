#download_volumes.py
#Copyright (c) 2020 Rachel Lea Ballantyne Draelos

#MIT License

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE

import io
import os
import time
import pickle
import requests
import pydicom
import warnings
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from requests_toolbelt.multipart import decoder

warnings.filterwarnings( 'ignore', '.*Unverified HTTPS.*' )

class DownloadData(object):
    def __init__(self, current_index, raw_accessions_path, save_path):
        """Download CT data in bulk and create a log file documenting the
        download process.
        
        <current_index> is an int for the row number to start from in the
            list of accessions.
        <raw_accessions_path> is the path to a | separated file with columns
            'MRN','AccessionNumber','Name', 'Date','Modality','Protocol', and
            'Study_UID'. It specifies the scans to be downloaded.
        <save_path> is a path to a directory where the downloaded CT scans
            will be saved."""
        self.current_index = current_index
        print('Starting from current_index',current_index)
        self.save_path = save_path
        
        self.r = get_token(verbose=True)
        self.error_count = 0
        self.initialize_logging_df() #init logging dfs and counters
        
        #Download data
        #The columns of raw_accessions_file are
        #'MRN','AccessionNumber','Name','Date','Modality','Protocol','Study_UID'
        myfiles = pd.read_csv(raw_accessions_path,sep='|',header=0,index_col=False,comment='#')
        self.read_and_save_everything_from_df(myfiles)

    # Initialization #----------------------------------------------------------
    def initialize_logging_df(self):
        self.logging_df = pd.DataFrame(np.empty((1,7),dtype='str'),
                        columns = ['CurrentIndex','AccessionNumber',
                                   'SeriesNumber','ImageType','SeriesDescription',
                                   'NumSlices','ChosenOne'])
        self.logging_df_name = datetime.datetime.today().strftime('%Y-%m-%d')+'_Logging_Dataframe.csv'
        self.logging_df_idx = 0
    
    # Read and Save scans key function #----------------------------------------
    def read_and_save_everything_from_df(self, accs_df):
        try:
            self.read_and_save(accs_df.loc[self.current_index:,:])
        except Exception as e:
            self.error_count+=1
            print(e)
            print('Error: waiting five minutes before restarting')
            time.sleep(300) #wait five minutes
            self.read_and_save(accs_df.loc[self.current_index:,:])
    
    # Read and Save helper functions #------------------------------------------
    def read_and_save(self, accs_df):
        total = accs_df.shape[0]
        for indexnum in accs_df.index.values.tolist():
            acc = accs_df.at[indexnum, 'AccessionNumber']
            self.r = get_token(verbose=False)
            t0 = time.time()
            self.get_one_series_custom(accession=acc, frames_thr=30)
            t1 = time.time()
            print('Finished ',acc,', index=',indexnum,' time=',round(t1-t0,2),'sec\n')
            self.current_index = indexnum
            self.error_count = max(0,self.error_count-1)
            
    def get_one_series_custom(self, accession, frames_thr):
        """Save all series for the specified accession number <accession>"""
        access_token = self.r['access_token']
        try:
            series_info = query_series_info(access_token, accession)
        except Exception as e:
            self.error_count+=1
            if self.error_count > 10:
                assert False, 'Quitting - too many errors'
            print(e,'\nfailed query_series_info on',accession)
            return(e)
        
        if series_info == []:
            self.error_count+=1
            if self.error_count > 10:
                assert False, 'Quitting - too many errors'
            print('failed - series is []',accession)
            return('fail')
        else:
            try:
                series_df = pd.DataFrame(series_info)
                #Keep only series with 'instances_number' greater than <frames_thr>:
                series_df = series_df[series_df['instances_number']>frames_thr]
                
                #get all possible scans for this accession:
                all_series_data = []
                for index, row in series_df.iterrows():
                    series_data = download_series(access_token, row['study_uid'], row['series_uid'] )
                    if series_data:
                        all_series_data.append(series_data)
                        
                #choose one scan to save:
                chosen_one, chosen_one_series_number, temp_logging_df = self.choose_scan_to_save(all_series_data)
                self.update_logging_df(accession, temp_logging_df)
                
                fname = os.path.join(self.save_path, row['accession']+'_'+str(chosen_one_series_number)+'.pkl' )
                of = open(fname, 'wb')
                pickle.dump(chosen_one, of)
                of.close()
                return('success')
                
            except Exception as e:
                self.error_count+=1
                if self.error_count > 10:
                    assert False, 'Quitting - too many errors'
                print(e,'\nfailed on',accession)
                return(e)

    def choose_scan_to_save(self, all_series_data):
        """Return the scan that should be saved.
        Variables:
        <all_series_data> is a list. Each element of the list is a sublist composed of
            pydicom.dataset.FileDataset objects in which each pydicom object is
            a DICOM for a single slice of a scan."""
        temp_logging_df = pd.DataFrame(np.empty((len(all_series_data),6),dtype='object'),
                        index = [x for x in range(len(all_series_data))],
                        columns = ['Data','SeriesNumber','SeriesDescription',
                                   'NumSlices','ImageType','ChosenOne'])
        
        for idx in range(len(all_series_data)):
            #data is itself a list of pydicom datasets
            data = all_series_data[idx]
            temp_logging_df.at[idx,'Data'] = data
            temp_logging_df.at[idx,'SeriesNumber'] = data[0]['0020','0011'].value #Series Number
            temp_logging_df.at[idx,'SeriesDescription'] = data[0]['008','103e'].value #Series Description
            temp_logging_df.at[idx,'NumSlices'] = len(data) #total number of slices
            #image type
            image_type = data[0]['008','008'].value #Image Type
            if 'ORIGINAL' in image_type:
                temp_logging_df.at[idx,'ImageType'] = 'ORIGINAL'
            elif ('DERIVED' in image_type) or ('SECONDARY' in image_type) or ('REFORMATTED' in image_type):
                temp_logging_df.at[idx,'ImageType'] = 'DERIVED'
            else:
                temp_logging_df.at[idx,'ImageType'] = '-'.join(list(image_type))
            temp_logging_df.at[idx,'ChosenOne'] = 'No'
        
        #Make selection
        keepers = temp_logging_df[temp_logging_df['ImageType']=='ORIGINAL'] #only keep ORIGINAL scans
        if keepers.shape[0]==0:
            return None, temp_logging_df
        #if we reach this point, there's at least one original scan
        #choose the original scan with the highest number of slices:
        max_slices = 0
        index_of_max_slices = None
        for keepidx in keepers.index.values.tolist():
            this_numslices = keepers.at[keepidx,'NumSlices']
            if this_numslices > max_slices:
                max_slices = this_numslices
                index_of_max_slices = keepidx
        
        #check if there was ever a situation where a non-original scan had
        #a greater number of slices than the original one you chose
        for tempidx in temp_logging_df.index.values.tolist():
            if temp_logging_df.at[tempidx,'NumSlices'] > max_slices:
                print('Non-Original Slice Exceeded Original Slices:',str(temp_logging_df.loc[tempidx,:]))
        
        #return chosen_one and temp_logging_df
        chosen_one = keepers.at[index_of_max_slices,'Data']
        chosen_one_series_number = keepers.at[index_of_max_slices,'SeriesNumber']
        temp_logging_df.at[index_of_max_slices,'ChosenOne'] = 'Yes'
        return chosen_one, chosen_one_series_number, temp_logging_df
                
    def update_logging_df(self, accession, temp_logging_df):
        """Update the overall logging df based on <temp_logging_df>
        and save the current version"""
        #Columns of self.logging_df are 'CurrentIndex','AccessionNumber',
        #'SeriesNumber','ImageType','SeriesDescription','NumSlices','ChosenOne'
        #Columns of temp_logging_df are 'Data','SeriesNumber','SeriesDescription',
        #'NumSlices','ImageType','ChosenOne'
        for idx in temp_logging_df.index.values.tolist():
            self.logging_df.at[self.logging_df_idx, 'CurrentIndex'] = self.current_index
            self.logging_df.at[self.logging_df_idx, 'AccessionNumber'] = accession
            self.logging_df.at[self.logging_df_idx, 'SeriesNumber'] = temp_logging_df.at[idx,'SeriesNumber']
            self.logging_df.at[self.logging_df_idx, 'ImageType'] = temp_logging_df.at[idx,'ImageType']
            self.logging_df.at[self.logging_df_idx, 'SeriesDescription'] = temp_logging_df.at[idx,'SeriesDescription']
            self.logging_df.at[self.logging_df_idx, 'NumSlices'] = temp_logging_df.at[idx,'NumSlices']
            self.logging_df.at[self.logging_df_idx, 'ChosenOne'] = temp_logging_df.at[idx,'ChosenOne']
            self.logging_df_idx+=1
        self.logging_df.to_csv(self.logging_df_name)
        
####################################
# Vendor Neutral Archive Functions #--------------------------------------------
####################################
def get_token(verbose=False):
    endpoint = 'SOME-ENDPOINT' #Not included in public repo
    client_id = 'SOME-CLIENT-ID' #Not included in public repo
    refresh_token = 'SOME-TOKEN' #Not included in public repo
    if verbose: print(client_id,'\n',refresh_token)
    r = requests.post( endpoint, data = { 'client_id':client_id, 'grant_type':'refresh_token', 'refresh_token':refresh_token }, verify=False )
    if verbose: print( 'request ok:', r.ok )
    return r.json()

def query_series_info(access_token, accession_number):
    """Pull information on the series specified by <accession_number>"""
    qendpoint = 'SOME-QENDPOINT' #Not included in public repo
    qhead = { 'Authorization':'token {}'.format( access_token ),'Accept':'application/json' }
    g = requests.get( qendpoint, params={'AccessionNumber':accession_number, 'Modality':'CT', 'offset':0 }, headers=qhead, verify=False )
    
    # parse
    series = []
    if ( g.ok and ( g.status_code == 200 ) ):
        len( g.json() )
        series_raw = g.json()
        
        for item in g.json():
            #Here are the possible keys you can get:
            #dict_keys(['00080060', '0008103E', '00081190', '0020000E',
            #'00200011', '00201209', '00080020', '00080030', '00080050',
            #'00080061', '00080090', '00100010', '00100020', '00100030',
            #'00100040', '0020000D', '00200010', '00201206', '00201208'])
            series_ = {}
            series_['study_uid'] = item.get('0020000D')['Value'][0]
            series_['series_uid'] = item.get('0020000E')['Value'][0]
            series_['accession'] = item.get('00080050')['Value'][0]
            series_['instances_number'] = item.get('00201209')['Value'][0]
            series_['series_description'] = item.get('0008103E')['Value'][0]
            series.append(series_)
    return series

def download_series(access_token, study_uid, series_uid):
    wendpoint = 'SOME-WENDPOINT/{study_instance_uid}/series/{series_instance_uid}'.format(  study_instance_uid=study_uid, series_instance_uid=series_uid )
    whead = {'Authorization': 'token {}'.format( access_token ),'Accept':'multipart/related; type=application/dicom'}
    h = requests.get( wendpoint, headers=whead, verify=False )
    ds = [];
    if ( h.ok and ( h.status_code == 200 ) ):
        mp_data = decoder.MultipartDecoder.from_response( h )
        ds = [ pydicom.dcmread( io.BytesIO( part.content ) ) for part in mp_data.parts ]
    return ds
