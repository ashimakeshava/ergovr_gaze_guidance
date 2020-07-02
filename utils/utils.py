import ProtobufTypes_pb2
import pandas as pd
import sys
from google.protobuf.json_format import MessageToDict
import numpy as np
import glob

def read_preprocess_save(data_path, save_path):
    '''
    read files from data_path, rename columns, and save in save_path
    Keyword arguments:
    data_path -- string path to .etd files
    save_path -- string path to save csv file
    Return: None
    '''
    protobuf_obj = ProtobufTypes_pb2.EyetrackingDataSet()
    files = glob.glob(f'{data_path}/subject2*.etd')
    samples_df = pd.DataFrame()
    shelf_df = pd.DataFrame()
    for fi in files:
        print(fi)
        try:
            with open(fi, "rb") as f:
                protobuf_obj.ParseFromString(f.read())
            dict_obj = MessageToDict(protobuf_obj)
            for nT, trial in enumerate(dict_obj['trials']):
                tmpdf = pd.io.json.json_normalize(data=trial['samples'])
    #             if fi.endswith('_2.etd'):
                tmpdf['trialID'] = (trial['metaData']['trialID']
                                    if 'trialID' in trial['metaData']
                                    else np.NaN
                                    )
                tmpdf['subjectID'] = dict_obj['subjectID']
                tmpdf['trialNum'] = nT
                tmpdf['subjectfileName'] = fi
                samples_df = pd.concat([samples_df, tmpdf], ignore_index=True, sort=False)
                tmpdf = pd.io.json.json_normalize(
                                data=trial['metaData']['initialConfiguration']['items']
                                )
                tmpdf['trialID'] = (trial['metaData']['trialID']
                                    if 'trialID' in trial['metaData']
                                    else np.NaN
                                    )
                tmpdf['subjectID'] = dict_obj['subjectID']
                tmpdf['subjectfileName'] = fi
                tmpdf['trialNum'] = nT
                shelf_df = pd.concat([shelf_df, tmpdf], ignore_index=True, sort=False)

        except FileNotFoundError:
            print(f'{fi} not found, moving on!')

    samples_df.columns = [col.replace('.','_') for col in samples_df.columns]
    samples_df['timestamp_dt'] = samples_df.timestamp
    samples_df = samples_df.query('timestamp>=0')
    samples_df.timestamp_dt = pd.to_datetime(samples_df.timestamp_dt, unit='s')
    shelf_df.fillna(value = {'shape':'Sphere', 'color':'Red',
                            'position.x':0, 'position.y':0, 'trialID':0},
                    inplace = True
                    )
    shelf_df.columns = [col.replace('.','_') for col in shelf_df.columns]
    #save dataframes
    samples_df.to_csv(f'{save_path}00_ET_samples_master.csv', index=False)
    shelf_df.to_csv(f'{save_path}00_ET_shelfData_master.csv', index=False)

def calculate_theta(samples_df):
    '''
    calculate horizontal and vertical angles for left, right, and combined
    eye movements in degrees
    Keyword arguments:
    samples_df: data frame with samples from all subjects
    Returns:
    samples_df dataframe with angles in degrees
    '''
    samples_df['left_eye_theta_h'] = np.arctan2(samples_df['leftEye_direction_x'],
                                                samples_df['leftEye_direction_z'])
    samples_df['left_eye_theta_v'] = np.arctan2(samples_df['leftEye_direction_y'],
                                        samples_df['leftEye_direction_z'])
    samples_df['right_eye_theta_h'] = np.arctan2(samples_df['rightEye_direction_x'],
                                                samples_df['rightEye_direction_z'])
    samples_df['right_eye_theta_v'] = np.arctan2(samples_df['rightEye_direction_y'],
                                                samples_df['rightEye_direction_z'])
    samples_df['combined_eye_theta_h'] = np.arctan2(samples_df['combinedEye_direction_x'],
                                                    samples_df['combinedEye_direction_z'])
    samples_df['combined_eye_theta_v'] = np.arctan2(samples_df['combinedEye_direction_y'],
                                                    samples_df['combinedEye_direction_z'])
    samples_df['head_theta_h'] = np.arctan2(samples_df['nosePointer_direction_x'],
                                            samples_df['nosePointer_direction_z'])
    samples_df['head_theta_v'] = np.arctan2(samples_df['nosePointer_direction_y'],
                                            samples_df['nosePointer_direction_z'])

    return samples_df

def calculate_EIH_theta(samples_df):
    '''
    calculate horizontal and vertical angles for left, right, and combined
    eye movements in degrees
    Keyword arguments:
    samples_df: data frame with samples from all subjects
    Returns:
    samples_df dataframe with angles in degrees
    '''
    samples_df['combined_eye_theta_h'] = np.arctan2(samples_df['EIH_dir_x'],
                                                    samples_df['EIH_dir_z'])
    samples_df['combined_eye_theta_v'] = np.arctan2(samples_df['EIH_dir_y'],
                                                    samples_df['EIH_dir_z'])
    samples_df['head_theta_h'] = np.arctan2(samples_df['nosePointer_direction_x'],
                                            samples_df['nosePointer_direction_z'])
    samples_df['head_theta_v'] = np.arctan2(samples_df['nosePointer_direction_y'],
                                            samples_df['nosePointer_direction_z'])

    return samples_df

def calculate_angular_velocity(samples_df):
    '''
    calculate horizontal and vertical angular velocities for left, right,
    and combined eye also for head movements in degrees
    Keyword arguments:
    samples_df: data frame with samples from all subjects
    Returns:
    samples_df dataframe with angular velocity in deg/s
    '''
    samples_df['left_eye_vel_h'] = (samples_df
                                    .groupby(['subjectID','subjectfileName','trialNum'])
                                    .left_eye_theta_h
                                    .apply(lambda x: x.diff()))/(samples_df
                                    .groupby(['subjectID','subjectfileName','trialNum'])
                                    .timestamp_dt
                                    .apply(lambda x: x.diff()/np.timedelta64(1, 's')))
    samples_df['left_eye_vel_h'] = samples_df['left_eye_vel_h']*180/np.pi

    samples_df['left_eye_vel_v'] = (samples_df
                                    .groupby(['subjectID','subjectfileName','trialNum'])
                                    .left_eye_theta_v
                                    .apply(lambda x: x.diff()))/(samples_df
                                    .groupby(['subjectID','subjectfileName','trialNum'])
                                    .timestamp_dt
                                    .apply(lambda x: x.diff()/np.timedelta64(1, 's')))
    samples_df['left_eye_vel_v'] = samples_df['left_eye_vel_v']*180/np.pi

    samples_df['right_eye_vel_h'] = (samples_df
                                    .groupby(['subjectID','subjectfileName','trialNum'])
                                    .right_eye_theta_h
                                    .apply(lambda x: x.diff()))/(samples_df
                                    .groupby(['subjectID','subjectfileName','trialNum'])
                                    .timestamp_dt
                                    .apply(lambda x: x.diff()/np.timedelta64(1, 's')))
    samples_df['right_eye_vel_h'] = samples_df['right_eye_vel_h']*180/np.pi

    samples_df['right_eye_vel_v'] = (samples_df
                                    .groupby(['subjectID','subjectfileName','trialNum'])
                                    .right_eye_theta_v
                                    .apply(lambda x: x.diff()))/(samples_df
                                    .groupby(['subjectID','subjectfileName','trialNum'])
                                    .timestamp_dt
                                    .apply(lambda x: x.diff()/np.timedelta64(1, 's')))
    samples_df['right_eye_vel_v'] = samples_df['right_eye_vel_v']*180/np.pi

    samples_df['combined_eye_vel_h'] = (samples_df
                                    .groupby(['subjectID','subjectfileName','trialNum'])
                                    .combined_eye_theta_h
                                    .apply(lambda x: x.diff()))/(samples_df
                                    .groupby(['subjectID','subjectfileName','trialNum'])
                                    .timestamp_dt
                                    .apply(lambda x: x.diff()/np.timedelta64(1, 's')))

    samples_df['combined_eye_vel_h'] = samples_df['combined_eye_vel_h']*180/np.pi

    samples_df['combined_eye_vel_v'] = (samples_df
                                    .groupby(['subjectID','subjectfileName','trialNum'])
                                    .combined_eye_theta_v
                                    .apply(lambda x: x.diff()))/(samples_df
                                    .groupby(['subjectID','subjectfileName','trialNum'])
                                    .timestamp_dt
                                    .apply(lambda x: x.diff()/np.timedelta64(1, 's')))

    samples_df['combined_eye_vel_v'] = samples_df['combined_eye_vel_v']*180/np.pi

    samples_df['head_vel_h'] = (samples_df
                                .groupby(['subjectID','subjectfileName','trialNum'])
                                .head_theta_h
                                .apply(lambda x: x.diff()))/(samples_df
                                .groupby(['subjectID','subjectfileName','trialNum'])
                                .timestamp_dt
                                .apply(lambda x: x.diff()/np.timedelta64(1, 's')))
    samples_df['head_vel_h'] = samples_df['head_vel_h']*180/np.pi

    samples_df['head_vel_v'] = (samples_df
                                    .groupby(['subjectID','subjectfileName','trialNum'])
                                    .head_theta_v
                                    .apply(lambda x: x.diff()))/(samples_df
                                    .groupby(['subjectID','subjectfileName','trialNum'])
                                    .timestamp_dt
                                    .apply(lambda x: x.diff()/np.timedelta64(1, 's')))
    samples_df['head_vel_v'] = samples_df['head_vel_v']*180/np.pi

    # get angular velocity by combining horizontal and vertical velocities
    samples_df['eye_angular_vel'] =np.sqrt(samples_df.combined_eye_vel_h**2 + samples_df.combined_eye_vel_v**2)
    samples_df['head_angular_vel'] =np.sqrt(samples_df.head_vel_h**2 + samples_df.head_vel_v**2)
    #make inf values NaN
    samples_df.combined_eye_vel_h = samples_df.combined_eye_vel_h.replace([np.inf, -np.inf], np.nan)
    samples_df.head_vel_h = samples_df.head_vel_h.replace([np.inf, -np.inf], np.nan)
    samples_df.combined_eye_vel_v = samples_df.combined_eye_vel_v.replace([np.inf, -np.inf], np.nan)
    samples_df.head_vel_v = samples_df.head_vel_v.replace([np.inf, -np.inf], np.nan)
    samples_df.eye_angular_vel = samples_df.eye_angular_vel.replace([np.inf, -np.inf], np.nan)
    samples_df.head_angular_vel = samples_df.head_angular_vel.replace([np.inf, -np.inf], np.nan)

    # drop NaN values
    samples_df = samples_df.dropna(subset=['combined_eye_vel_h','head_vel_h',\
                                       'combined_eye_vel_v', 'head_vel_v',\
                                      'eye_angular_vel', 'head_angular_vel'])

    return samples_df

def calculate_EIH_angular_velocity(samples_df):
    '''
    calculate horizontal and vertical angular velocities for left, right,
    and combined eye also for head movements in degrees
    Keyword arguments:
    samples_df: data frame with samples from all subjects
    Returns:
    samples_df dataframe with angular velocity in deg/s
    '''
    samples_df['combined_eye_vel_h'] = (samples_df
                                    .groupby(['subjectID','subjectfileName','trialNum'])
                                    .combined_eye_theta_h
                                    .apply(lambda x: x.diff()))/(samples_df
                                    .groupby(['subjectID','subjectfileName','trialNum'])
                                    .timestamp_dt
                                    .apply(lambda x: x.diff()/np.timedelta64(1, 's')))

    samples_df['combined_eye_vel_h'] = samples_df['combined_eye_vel_h']*180/np.pi

    samples_df['combined_eye_vel_v'] = (samples_df
                                    .groupby(['subjectID','subjectfileName','trialNum'])
                                    .combined_eye_theta_v
                                    .apply(lambda x: x.diff()))/(samples_df
                                    .groupby(['subjectID','subjectfileName','trialNum'])
                                    .timestamp_dt
                                    .apply(lambda x: x.diff()/np.timedelta64(1, 's')))

    samples_df['combined_eye_vel_v'] = samples_df['combined_eye_vel_v']*180/np.pi

    samples_df['head_vel_h'] = (samples_df
                                .groupby(['subjectID','subjectfileName','trialNum'])
                                .head_theta_h
                                .apply(lambda x: x.diff()))/(samples_df
                                .groupby(['subjectID','subjectfileName','trialNum'])
                                .timestamp_dt
                                .apply(lambda x: x.diff()/np.timedelta64(1, 's')))
    samples_df['head_vel_h'] = samples_df['head_vel_h']*180/np.pi

    samples_df['head_vel_v'] = (samples_df
                                    .groupby(['subjectID','subjectfileName','trialNum'])
                                    .head_theta_v
                                    .apply(lambda x: x.diff()))/(samples_df
                                    .groupby(['subjectID','subjectfileName','trialNum'])
                                    .timestamp_dt
                                    .apply(lambda x: x.diff()/np.timedelta64(1, 's')))
    samples_df['head_vel_v'] = samples_df['head_vel_v']*180/np.pi

    # get angular velocity by combining horizontal and vertical velocities
    samples_df['eye_angular_vel'] =np.sqrt(samples_df.combined_eye_vel_h**2 + samples_df.combined_eye_vel_v**2)
    samples_df['head_angular_vel'] =np.sqrt(samples_df.head_vel_h**2 + samples_df.head_vel_v**2)
    #make inf values NaN
    samples_df.combined_eye_vel_h = samples_df.combined_eye_vel_h.replace([np.inf, -np.inf], np.nan)
    samples_df.head_vel_h = samples_df.head_vel_h.replace([np.inf, -np.inf], np.nan)
    samples_df.combined_eye_vel_v = samples_df.combined_eye_vel_v.replace([np.inf, -np.inf], np.nan)
    samples_df.head_vel_v = samples_df.head_vel_v.replace([np.inf, -np.inf], np.nan)
    samples_df.eye_angular_vel = samples_df.eye_angular_vel.replace([np.inf, -np.inf], np.nan)
    samples_df.head_angular_vel = samples_df.head_angular_vel.replace([np.inf, -np.inf], np.nan)

    # drop NaN values
    samples_df = samples_df.dropna(subset=['combined_eye_vel_h','head_vel_h',\
                                       'combined_eye_vel_v', 'head_vel_v',\
                                      'eye_angular_vel', 'head_angular_vel'])

    return samples_df

def calculate_angular_acceleration(samples_df):
    '''
    calculate horizontal and vertical angular acceleration for left, right, and combined
    eye movements in degrees
    Keyword arguments:
    samples_df: data frame with samples from all subjects
    Returns:
    samples_df dataframe with angular acceleration in deg/s^2
    '''
    samples_df['left_eye_acc_h'] = (samples_df
                                    .groupby(['subjectID','subjectfileName','trialNum'])
                                    .left_eye_vel_h
                                    .apply(lambda x: x.diff()))/(samples_df
                                    .groupby(['subjectID','subjectfileName','trialNum'])
                                    .timestamp_dt
                                    .apply(lambda x: x.diff()/np.timedelta64(1, 's')))
    samples_df['left_eye_acc_v'] = (samples_df
                                    .groupby(['subjectID','subjectfileName','trialNum'])
                                    .left_eye_vel_v
                                    .apply(lambda x: x.diff()))/(samples_df
                                    .groupby(['subjectID','subjectfileName','trialNum'])
                                    .timestamp_dt
                                    .apply(lambda x: x.diff()/np.timedelta64(1, 's')))
    samples_df['right_eye_acc_h'] = (samples_df
                                    .groupby(['subjectID','subjectfileName','trialNum'])
                                    .right_eye_vel_h
                                    .apply(lambda x: x.diff()))/(samples_df
                                    .groupby(['subjectID','subjectfileName','trialNum'])
                                    .timestamp_dt
                                    .apply(lambda x: x.diff()/np.timedelta64(1, 's')))
    samples_df['right_eye_acc_v'] = (samples_df
                                    .groupby(['subjectID','subjectfileName','trialNum'])
                                    .right_eye_vel_v
                                    .apply(lambda x: x.diff()))/(samples_df
                                    .groupby(['subjectID','subjectfileName','trialNum'])
                                    .timestamp_dt
                                    .apply(lambda x: x.diff()/np.timedelta64(1, 's')))
    samples_df['combined_eye_acc_h'] = (samples_df
                                    .groupby(['subjectID','subjectfileName','trialNum'])
                                    .combined_eye_vel_h
                                    .apply(lambda x: x.diff()))/(samples_df
                                    .groupby(['subjectID','subjectfileName','trialNum'])
                                    .timestamp_dt
                                    .apply(lambda x: x.diff()/np.timedelta64(1, 's')))
    samples_df['combined_eye_acc_v'] = (samples_df
                                    .groupby(['subjectID','subjectfileName','trialNum'])
                                    .combined_eye_vel_v
                                    .apply(lambda x: x.diff()))/(samples_df
                                    .groupby(['subjectID','subjectfileName','trialNum'])
                                    .timestamp_dt
                                    .apply(lambda x: x.diff()/np.timedelta64(1, 's')))
    samples_df['head_acc_h'] = (samples_df
                                .groupby(['subjectID','subjectfileName','trialNum'])
                                .head_vel_h
                                .apply(lambda x: x.diff()))/(samples_df
                                .groupby(['subjectID','subjectfileName','trialNum'])
                                .timestamp_dt
                                .apply(lambda x: x.diff()/np.timedelta64(1, 's')))
    samples_df['head_acc_v'] = (samples_df
                                    .groupby(['subjectID','subjectfileName','trialNum'])
                                    .head_vel_v
                                    .apply(lambda x: x.diff()))/(samples_df
                                    .groupby(['subjectID','subjectfileName','trialNum'])
                                    .timestamp_dt
                                    .apply(lambda x: x.diff()/np.timedelta64(1, 's')))

    samples_df.combined_eye_acc_h = samples_df.combined_eye_acc_h.replace([np.inf, -np.inf], np.nan)
    samples_df.head_acc_h = samples_df.head_acc_h.replace([np.inf, -np.inf], np.nan)
    samples_df.combined_eye_acc_v = samples_df.combined_eye_acc_v.replace([np.inf, -np.inf], np.nan)
    samples_df.head_acc_v = samples_df.head_acc_v.replace([np.inf, -np.inf], np.nan)

    samples_df = samples_df.dropna(subset=['combined_eye_acc_h', 'head_acc_h',
                                       'combined_eye_acc_v', 'head_acc_v'])

    return samples_df

def calculate_EIH_angular_acceleration(samples_df):
    '''
    calculate horizontal and vertical angular acceleration for left, right, and combined
    eye movements in degrees
    Keyword arguments:
    samples_df: data frame with samples from all subjects
    Returns:
    samples_df dataframe with angular acceleration in deg/s^2
    '''
    samples_df['combined_eye_acc_h'] = (samples_df
                                    .groupby(['subjectID','subjectfileName','trialNum'])
                                    .combined_eye_vel_h
                                    .apply(lambda x: x.diff()))/(samples_df
                                    .groupby(['subjectID','subjectfileName','trialNum'])
                                    .timestamp_dt
                                    .apply(lambda x: x.diff()/np.timedelta64(1, 's')))
    samples_df['combined_eye_acc_v'] = (samples_df
                                    .groupby(['subjectID','subjectfileName','trialNum'])
                                    .combined_eye_vel_v
                                    .apply(lambda x: x.diff()))/(samples_df
                                    .groupby(['subjectID','subjectfileName','trialNum'])
                                    .timestamp_dt
                                    .apply(lambda x: x.diff()/np.timedelta64(1, 's')))
    samples_df['head_acc_h'] = (samples_df
                                .groupby(['subjectID','subjectfileName','trialNum'])
                                .head_vel_h
                                .apply(lambda x: x.diff()))/(samples_df
                                .groupby(['subjectID','subjectfileName','trialNum'])
                                .timestamp_dt
                                .apply(lambda x: x.diff()/np.timedelta64(1, 's')))
    samples_df['head_acc_v'] = (samples_df
                                    .groupby(['subjectID','subjectfileName','trialNum'])
                                    .head_vel_v
                                    .apply(lambda x: x.diff()))/(samples_df
                                    .groupby(['subjectID','subjectfileName','trialNum'])
                                    .timestamp_dt
                                    .apply(lambda x: x.diff()/np.timedelta64(1, 's')))

    samples_df.combined_eye_acc_h = samples_df.combined_eye_acc_h.replace([np.inf, -np.inf], np.nan)
    samples_df.head_acc_h = samples_df.head_acc_h.replace([np.inf, -np.inf], np.nan)
    samples_df.combined_eye_acc_v = samples_df.combined_eye_acc_v.replace([np.inf, -np.inf], np.nan)
    samples_df.head_acc_v = samples_df.head_acc_v.replace([np.inf, -np.inf], np.nan)

    samples_df = samples_df.dropna(subset=['combined_eye_acc_h', 'head_acc_h',
                                       'combined_eye_acc_v', 'head_acc_v'])

    return samples_df

def downsample_data(samples_df, sampling_rate=75):
    '''
    downsamples dataframe to given sampling_rate
    '''
    td = pd.Timedelta(0.013333, 's')
    samples_df = (samples_df
              .set_index('timestamp_dt')
              .groupby(['subjectID', 'subjectfileName', 'trialNum'])
              .resample('0.013333S',loffset=td, convention='s').fillna('ffill')
             )
    samples_df.drop(columns=['subjectID', 'subjectfileName', 'trialNum'],inplace=True)
    samples_df = samples_df.reset_index()
    return samples_df

def med_filt(x,samples=3):
    return pd.Series(ss.medfilt(x,samples))

def simple_mad(angular_vel, thresh = 3.5):
#     th_1 = np.median(angular_vel)
    if len(angular_vel.shape) == 1:
        angular_vel = angular_vel[:,None]
    median = np.median(angular_vel)
    diff = (angular_vel - median)**2
    diff = np.sqrt(diff)
#     print(diff)
    med_abs_deviation = np.median(diff)
    saccade_thresh = median + thresh*med_abs_deviation
    return saccade_thresh

def at_mad(angular_vel, th_0 = 200):
    threshs = []
    if len(angular_vel.shape) == 1:
        angular_vel = angular_vel[:,None]
    while True:
        threshs.append(th_0)
        angular_vel = angular_vel[angular_vel < th_0]
        median = np.median(angular_vel)
        diff = (angular_vel - median)**2
        diff = np.sqrt(diff)
        med_abs_deviation = np.median(diff)
        th_1 = median + 3*1.48*med_abs_deviation
#         print(th_0, th_1)
        if (th_0 - th_1)>1:
            th_0 = th_1
        else:
            saccade_thresh = th_1
            threshs.append(saccade_thresh)
            break
    return saccade_thresh

def get_fixation_samples(samples_df):
    samples_df['isHeadStable'] = False
    samples_df['isHeadStable'] = (samples_df
                           .groupby(['subjectID', 'subjectfileName', 'trialNum'],
                                    as_index=False)
                           .head_angular_vel
                           .transform(lambda x: x < at_mad(x, 100))
                          )
    samples_df['isFixV'] = False
    samples_df['isFixV'] = (samples_df
                           .groupby(['subjectID', 'subjectfileName', 'trialNum'],
                                    as_index=False)
                           .eye_angular_vel
                           .transform(lambda x: x < at_mad(x))
                          )

    return samples_df

def get_fixation_duration(samples_df):
    def calculate_fixation_duration(samples_df):
        s_df_copy = samples_df.copy()
        s_df_copy.set_index('timestamp_dt', inplace=True)
        s_df_copy['fix_duration'] = (s_df_copy
                                      # .query('isHeadStable==1')
                                      .groupby(['subjectID', 'subjectfileName',
                                                'trialNum'],
                                               as_index=False
                                               ).isFixV
                                      .apply(lambda x:
                                             x.groupby((x != x.shift()).cumsum())
                                             .transform(lambda x:
                                                        (x.index[-1] - x.index[0]
                                                        )/np.timedelta64(1, 's')
                                                       )
                                            )
                                     ).reset_index().set_index('timestamp_dt').isFixV
        s_df_copy = s_df_copy.reset_index()
        return s_df_copy

    samples_df = calculate_fixation_duration(samples_df)
    # print(samples_df.head())
    ## if saccade less that 40ms make it part of fixation
    idx = samples_df.query('isFixV==0 and fix_duration<=0.03 and fix_duration>0 ').index
    samples_df.loc[idx,'isFixV'] = True
    # samples_df.loc[samples_df.query('isFixV==1 and fix_duration<0.1').index, 'isFixV'] = 0
    # recalculate fixation durations
    samples_df = calculate_fixation_duration(samples_df)
    samples_df['isOutlierFix'] = (samples_df
                              .query('isFixV == 1 and fix_duration != 0')
#                               .groupby(['subjectID', 'subjectfileName', 'trialNum'], as_index=False)
                              .fix_duration
                              .transform(lambda x: x > simple_mad(x, 3))
                             )

    samples_df['isOutlierSac'] = (samples_df
                                  .query('isFixV == 0 and fix_duration != 0')
    #                               .groupby(['subjectID', 'subjectfileName', 'trialNum'], as_index=False)
                                  .fix_duration
                                  .transform(lambda x: x > simple_mad(x, 3))
                                 )
    samples_df.isOutlierFix.fillna(False, inplace=True)
    samples_df.isOutlierSac.fillna(False, inplace=True)

    samples_df.set_index('timestamp_dt', inplace=True)
    samples_df['fix_onset'] = (samples_df
                           .groupby(['subjectID', 'subjectfileName',
                                     'trialNum'], as_index=False).isFixV
                              .apply(lambda x:
                                     x.groupby((x != x.shift()).cumsum())
                                     .transform(lambda x: x.index[0])
                                    )
                             ).reset_index().set_index('timestamp_dt').isFixV

    samples_df['fix_stop'] = (samples_df
                           .groupby(['subjectID', 'subjectfileName',
                                     'trialNum'], as_index=False).isFixV
                              .apply(lambda x:
                                     x.groupby((x != x.shift()).cumsum())
                                     .transform(lambda x: x.index[-1])
                                    )
                             ).reset_index().set_index('timestamp_dt').isFixV
    samples_df = samples_df.reset_index()

    samples_df['fix_onset_bool'] = False
    idx = samples_df.query('timestamp_dt == fix_onset and fix_duration>0').index
    samples_df.loc[idx, 'fix_onset_bool'] = True

    samples_df['fix_stop_bool'] = False
    idx = samples_df.query('timestamp_dt == fix_stop and fix_duration>0').index
    samples_df.loc[idx, 'fix_stop_bool'] = True

    return samples_df

def get_grasp_info(samples_df):
    samples_df.drop(columns=['timestamp',
       'leftEye_position_x', 'leftEye_position_y', 'leftEye_position_z',
       'leftEye_direction_x', 'leftEye_direction_y', 'leftEye_direction_z',
       'leftEye_raycastHitObject', 'leftEye_raycastHitLocation_x',
       'leftEye_raycastHitLocation_y', 'leftEye_raycastHitLocation_z',
       'rightEye_position_x', 'rightEye_position_y', 'rightEye_position_z',
       'rightEye_direction_x', 'rightEye_direction_y','rightEye_direction_z',
        'rightEye_raycastHitObject','rightEye_raycastHitLocation_x',
        'rightEye_raycastHitLocation_y','rightEye_raycastHitLocation_z',
       'head_theta_h', 'head_theta_v', 'head_vel_h',
       'head_vel_v', 'head_acc_h', 'head_acc_v',
        ], inplace=True)
    samples_df.set_index('timestamp_dt', inplace=True)
    samples_df[['grasp_onset','grasp_stop','grasp_duration']] = (samples_df
                 .groupby(['subjectID', 'subjectfileName', 'trialNum'], as_index=False)
                 .handData_graspedObject
                 .apply(lambda x: x.groupby((x != x.shift()).cumsum())
                         .transform(lambda x: [x.index[0],
                         x.index[-1],
                         (x.index[-1] - x.index[0])/np.timedelta64(1,'s')
                         ] )
                        )
    ).reset_index().set_index('timestamp_dt').handData_graspedObject

    # samples_df['grasp_stop'] = (samples_df
    #             .groupby(['subjectID', 'subjectfileName', 'trialNum'], as_index=False)
    #              .handData_graspedObject
    #              .apply(lambda x: x.groupby((x != x.shift()).cumsum())
    #                      .transform(lambda x: x.index[-1])
    #                     )
    # ).reset_index().set_index('timestamp_dt').handData_graspedObject
    #
    # samples_df['grasp_duration'] = (samples_df
    #              .groupby(['subjectID', 'subjectfileName', 'trialNum'], as_index=False)
    #              .handData_graspedObject
    #              .apply(lambda x: x.groupby((x != x.shift()).cumsum())
    #                      .transform(
    #                      lambda x:
    #                         (x.index[-1] - x.index[0])/np.timedelta64(1,'s')
    #                      )
    #                     )
    # ).reset_index().set_index('timestamp_dt').handData_graspedObject

    samples_df = samples_df.reset_index()

    samples_df['grasp_onset_bool'] = False
    idx = samples_df.query('timestamp_dt == grasp_onset and grasp_duration>0').index
    samples_df.loc[idx, 'grasp_onset_bool'] = True
    samples_df['grasp_end_bool'] = False
    idx = samples_df.query('timestamp_dt == grasp_stop and grasp_duration>0').index
    samples_df.loc[idx, 'grasp_end_bool'] = True

    return samples_df

def get_shelf_centers(meta_data_path):
    files = glob.glob(f'{meta_data_path}/*_pos.csv')
    shelf_centers = pd.DataFrame()
    for f in files:
        tmpdf = pd.read_csv(f, sep=';')
        tmpdf['object'] = f.split('\\')[1].split('_')[0]
        shelf_centers = pd.concat([shelf_centers, tmpdf], ignore_index=True)

    shelf_centers.gridPosition = shelf_centers.gridPosition.str.strip('()')
    shelf_centers[['pos_x', 'pos_y']] = shelf_centers.gridPosition.str.split(',', expand=True)
    shelf_centers['world position'] = shelf_centers['world position'].str.strip('()')

    shelf_centers[['center_x', 'center_y', 'center_z']] = shelf_centers['world position'].str.split(' ', expand=True)
    shelf_centers.center_x = shelf_centers.center_x.str.replace(',', '.').str.strip('.')
    shelf_centers.center_y = shelf_centers.center_y.str.replace(',', '.').str.strip('.')
    shelf_centers.center_z = shelf_centers.center_z.str.replace(',', '.').str.strip('.')
    shelf_centers.drop(columns=['gridPosition', 'world position'], inplace=True)
    shelf_centers = shelf_centers.astype({'pos_x':'int32',
                                          'pos_y':'int32',
                                          'center_x':'float64',
                                          'center_y':'float64',
                                          'center_z':'float64'})

    shelf_centers.pos_x = shelf_centers.pos_x + 1
    shelf_centers.pos_y = shelf_centers.pos_y + 1
    shelf_centers['shelfID'] = shelf_centers.pos_y.astype(str)+'_'+shelf_centers.pos_x.astype(str)
    shelf_centers.object = shelf_centers.object.map({'cube':'Cube',
                                                     'sphere':'Sphere',
                                                     'cylinder':'Cylinder',
                                                     'pyramid':'Tetraeder'})
    return shelf_centers

def getShelfLoc(grasp, shelf_centers):
    grasp_object = grasp.handData_graspedObject.split('_')[0]
    #     print(grasp_object)
    shelf_centers = shelf_centers.query('object == @grasp_object')
    shelf_centers['dist'] = (np.sqrt(
        (grasp.handData_contactPoint_x - shelf_centers.center_x)**2 +
        (grasp.handData_contactPoint_y - shelf_centers.center_y)**2 +
        (grasp.handData_contactPoint_z - shelf_centers.center_z)**2
    ))
    return shelf_centers.loc[shelf_centers.dist.idxmin(),'shelfID']

def get_pickup_dropoff_loc(samples_df, meta_data_path):
    samples_df['grasp_onset_bool'] = False
    idx = samples_df.query('timestamp_dt == grasp_onset and grasp_duration>0').index
    samples_df.loc[idx, 'grasp_onset_bool'] = True
    samples_df['grasp_end_bool'] = False
    idx = samples_df.query('timestamp_dt == grasp_stop and grasp_duration>0').index
    samples_df.loc[idx, 'grasp_end_bool'] = True

    shelf_centers = get_shelf_centers(meta_data_path)
    samples_df['pickup_location'] = (samples_df
                             .query('grasp_onset_bool==1')
                             .apply(lambda x: getShelfLoc(x,shelf_centers), axis=1)
                            )
    samples_df['drop_location'] = (samples_df
                            .query('grasp_end_bool==1')
                            .apply(lambda x: getShelfLoc(x, shelf_centers), axis=1)
                            )

    return samples_df

def get_epoch_grasp_on(grp, objs_dict, shelf_dict, offset=1):
    grp = grp.set_index('timestamp_dt')
    grp['eyeHit'] = grp['eyeHit'].map(objs_dict)
    grp['grasp'] = grp['grasp'].map(objs_dict)
    grp['eye_shelfHits'] = grp['eye_shelfHits'].map(shelf_dict)
    grp['pickup_location'] = grp['pickup_location'].map(shelf_dict)
#     grp['headHit'] = grp['headHit'].map(objs_dict)
    graspNum = 0
    offset = pd.Timedelta(offset, 's')
    epoched = pd.DataFrame()
    for i, row in grp.iterrows():
        if row.grasp_onset_bool == True:
            graspNum = graspNum + 1
            tmp = grp.loc[i-offset:i+pd.Timedelta(1, 's')]
            tmp['targetObjectFix'] = tmp.eyeHit == row.grasp
            tmp['targetPickUpShelfFix'] = tmp.eye_shelfHits == row.pickup_location
            tmp['graspNum'] = graspNum
            tmp['time'] = (tmp.index - i)/np.timedelta64(1,'s')
            tmp['trialID'] = row.trialID
            epoched = pd.concat([epoched, tmp], ignore_index=True)
    return epoched
