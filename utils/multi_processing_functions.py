import pandas as pd
import numpy as np
from scipy import spatial
from functools import partial


def shelf_distance(shelf_a, shelf_b):
    if pd.isna(shelf_a) or pd.isna(shelf_b):
        return np.nan
    shelf_a_arr = np.array(list(map(int, shelf_a.split('_'))))
    shelf_b_arr = np.array(list(map(int, shelf_b.split('_'))))
    return spatial.distance.cityblock(shelf_a_arr, shelf_b_arr)


def get_epoch_grasp_on(grp, offset_start=0, offset_stop=-4, bin_size=0.25):
    grp_cols = ['subject_id', 'trial_num', 'trial_type', 'grasp_num']
    sample_df, name = grp
    sample_df = sample_df.sort_values(by='timestamp_dt')

    grasp_times = (
        sample_df.query('grasp_onset_bool == 1')
        [['pickup_location', 'grasp', 'timestamp_dt']]
        .rename(columns=dict(
            timestamp_dt='on_time',
            pickup_location='on_loc',
            grasp='grasp_object',
        ))
    )
    grasp_times['off_time'] = sample_df.query('grasp_end_bool == 1').timestamp_dt.values
    grasp_times['off_loc'] = sample_df.query('grasp_end_bool == 1').drop_location.values
    grasp_times = grasp_times.sort_values(by='on_time').reset_index(drop=True)

    offset_start_pd = pd.Timedelta(offset_start, 's')
    offset_stop_pd = pd.Timedelta(offset_stop, 's')
    windows_df = pd.concat(
        grasp_times
        .apply(
            lambda row: (
                sample_df.loc[
                    # filter timestamps between window based on offsets
                    sample_df.timestamp_dt.between(
                        row.on_time + offset_start_pd, row.on_time + offset_stop_pd
                    ),

                    # columns from sample_df required for each epoch
                    ['trial_type', 'timestamp_dt', 'is_fixation', 'eye_hit',
                     'eye_shelf_hit',]
                ]
                .pipe(lambda df: df.assign(

                    # whether fixation is on grasped object
                    # when the fixation is on `Other` then return np.nan else indicate if the
                    # fixation is on the grasped object or not
                    target_object_fix=df.eye_hit.apply(
                        lambda s: (
                            np.nan if s == 'Other'
                            else s == row.grasp_object if not pd.isnull(s)
                            else np.nan
                            )
                    ),


                    # whether fixation is on target shelf
                    # when the fixation is on any object return np.nan else indicate if the
                    # fixation is on the target shelf where the grasped object is placed
                    target_shelf_fix=df[['eye_hit', 'eye_shelf_hit']].apply(
                        lambda s_row: (
                            np.nan if s_row.eye_hit != 'Other' and not pd.isnull(s_row.eye_hit)
                            else s_row.eye_shelf_hit == row.off_loc
                            if not pd.isnull(s_row.eye_shelf_hit) and not pd.isnull(s_row.eye_hit)
                            else np.nan
                            ),
                        axis=1
                    ),

                    non_target_object_fix=df.eye_hit.apply(
                        lambda s: (
                            np.nan if s == 'Other'
                            else s != row.grasp_object
                            if s != row.grasp_object and not pd.isnull(s)
                            else False
                            if s == row.grasp_object and not pd.isnull(s)
                            else np.nan
                        )
                    ),

                    non_target_shelf_fix=df[['eye_hit', 'eye_shelf_hit']].apply(
                        lambda s_row: (
                            np.nan if s_row.eye_hit != 'Other' and not pd.isnull(s_row.eye_hit)
                            else s_row.eye_shelf_hit != row.off_loc
                            if not pd.isnull(s_row.eye_shelf_hit) and not pd.isnull(s_row.eye_hit)
                            else np.nan
                        ),
                        axis=1
                    ),

                    non_target_object_same_color=df.eye_hit.apply(
                        lambda s: (
                            np.nan if s == 'Other' and not pd.isnull(s)
                            else s.split('_')[1] == row.grasp_object.split('_')[1]
                            if s != row.grasp_object and not pd.isnull(s)
                            else False
                            if s == row.grasp_object and not pd.isnull(s)
                            else np.nan
                        )
                    ),

                    non_target_object_same_shape=df.eye_hit.apply(
                        lambda s: (
                            np.nan if s == 'Other' and not pd.isnull(s)
                            else s.split('_')[0] == row.grasp_object.split('_')[0]
                            if s != row.grasp_object and not pd.isnull(s)
                            else False
                            if s == row.grasp_object and not pd.isnull(s)
                            else np.nan
                        )
                    ),

                    # grasp count in the trial
                    grasp_num=row.name,
                    grasp_object=row.grasp_object,

                    grasp_time=row.on_time,
                    drop_time=row.off_time,

                    drop_location=row.off_loc,
                    pickup_location=row.on_loc,

                    # manhattan distance of shelf fixations from the pickup location
                    proximity_pick=df.eye_shelf_hit.apply(
                        lambda s_hit : shelf_distance(s_hit, row.on_loc)
                    ),

                    # manhattan distance of shelf fixations from the drop location
                    proximity_drop=df.eye_shelf_hit.apply(
                        lambda s_hit : shelf_distance(s_hit, row.off_loc)
                    ),

                    # time since the start of the trial
                    sample_time=(df.timestamp_dt - row.on_time) / np.timedelta64(1, 's'),

                    subject_id=name[0],
                    trial_num=name[1]
                ))
                .pipe(lambda df: df.assign(
                    time_bin=pd.cut(
                        df.sample_time,
                        bins=np.arange(offset_start, offset_stop + bin_size, bin_size),
                        labels=np.arange(
                            offset_start + bin_size,
                            offset_stop + bin_size,
                            bin_size
                        ),
                    )
                ))
            ),
            axis=1,
        )
        .values
    ).set_index(grp_cols)
    return windows_df


def get_avg_fixations(grp):
    grp_cols = ['subject_id', 'trial_num', 'trial_type', 'time_bin']
    grasp_df, name = grp
    fix_prop_df = (
        grasp_df
        .groupby(grp_cols)
        .agg({
            'target_object_fix': [np.nansum, 'size'],
            'target_shelf_fix': [np.nansum, 'size'],
            'non_target_object_fix': np.nansum,
            'non_target_object_same_shape': np.nansum,
            'non_target_object_same_color': np.nansum,
            'non_target_shelf_fix': np.nansum,
            'proximity_pick': 'mean',
            'proximity_drop': 'mean',
        })
    )
    fix_prop_df.columns = [
        'target_object_fix_count', 'total_object_fix_count',
        'target_shelf_fix_count', 'total_shelf_fix_count',
        'non_target_object_fix_count',
        'non_target_object_same_shape_count',
        'non_target_object_same_color_count',
        'non_target_shelf_fix_count',
        'proximity_pick',
        'proximity_drop',
    ]
    fix_prop_df['target_object_fix_prop'] = (
        fix_prop_df.target_object_fix_count / fix_prop_df.total_object_fix_count
    )
    fix_prop_df['target_shelf_fix_prop'] = (
        fix_prop_df.target_shelf_fix_count / fix_prop_df.total_shelf_fix_count
    )
    fix_prop_df['non_target_object_fix_prop'] = (
        fix_prop_df.non_target_object_fix_count / fix_prop_df.total_object_fix_count
    )
    fix_prop_df['non_target_shelf_fix_prop'] = (
        fix_prop_df.non_target_shelf_fix_count / fix_prop_df.total_shelf_fix_count
    )
    fix_prop_df['non_target_object_same_feature_fix_prop'] = (
        (fix_prop_df.non_target_object_same_shape_count
        + fix_prop_df.non_target_object_same_color_count)
        / fix_prop_df.total_object_fix_count
    )
    fix_prop_df['non_target_object_diff_feature_fix_prop'] = (
        (fix_prop_df.non_target_object_fix_count
        - fix_prop_df.non_target_object_same_shape_count
        - fix_prop_df.non_target_object_same_color_count)
        / fix_prop_df.total_object_fix_count
    )
    # fix_prop_df['non_target_object_same_color_fix_prop'] = (
    #     fix_prop_df.non_target_object_same_color_count / fix_prop_df.total_object_fix_count
    # )

    return fix_prop_df


def get_epoch_grasp_between(grp, offset_start=2, offset_stop=-2):
    grp_cols = ['subject_id', 'trial_num', 'trial_type', 'grasp_num']
    sample_df, name = grp
    sample_df = sample_df.sort_values(by='timestamp_dt')

    grasp_times = (
        sample_df.query('grasp_onset_bool == 1')
        [['grasp', 'timestamp_dt']]
        .rename(columns=dict(
            timestamp_dt='current_grasp_time',
            grasp='current_grasp_object',
        ))
    )
    grasp_times['next_grasp_time'] = grasp_times.current_grasp_time.shift(-1)
    grasp_times['next_grasp_object'] = grasp_times.current_grasp_object.shift(-1)
    grasp_times = grasp_times.reset_index(drop=True)

    offset_start = pd.Timedelta(offset_start, 's')
    offset_stop = pd.Timedelta(offset_stop, 's')

    if len(grasp_times) <= 1:
        return pd.DataFrame()

    windows_df = pd.concat(

        grasp_times
        # drop the last row
        .head(-1)

        .apply(
            lambda row: (
                sample_df.loc[
                    # filter timestamps between window based on offsets
                    sample_df.timestamp_dt.between(
                        row.current_grasp_time + offset_start,
                        row.next_grasp_time + offset_stop,
                    ),

                    # columns from sample_df required for each epoch
                    ['trial_type', 'timestamp_dt', 'is_fixation', 'eye_hit',
                     'eye_shelf_hit',]
                ]

                .pipe(lambda df: df.assign(

                    # grasp count in the trial
                    grasp_num=row.name,

                    current_grasp_object=row.current_grasp_object,
                    next_grasp_object=row.next_grasp_object,

                    # time since the start of the trial
                    sample_time=(
                        (df.timestamp_dt - (row.current_grasp_time + offset_start))
                        / np.timedelta64(1, 's')
                    ),

                    subject_id=name[0],
                    trial_num=name[1]
                ))

                .groupby(grp_cols + ['current_grasp_object', 'next_grasp_object', 'eye_hit'])
                .size()
                .to_frame('hit_counts')
            ),
            axis=1,
        )
        .values
    )
    return windows_df


def expanding_pipe_df(df, window_function, min_periods=0):
    return pd.concat(
        [
            df.iloc[0:i].pipe(window_function)
            if i > min_periods else None
            for i in range(1, len(df) + 1)
        ],
    )


def compute_lookahead_distance(grasp_df, eye_hits_df):

    def grasp_diff(eye_hit_object, grasp_window):
        if eye_hit_object == 'Other':
            return np.nan

        grasped_df = grasp_window.query('current_grasp_object == @eye_hit_object')

        if len(grasped_df) == 0:
            return np.nan

        return grasped_df.tail(1).grasp_num.values[0]


    # if len(grasp_df) <= 1:
    #     return pd.DataFrame()

    current_grasp = grasp_df.tail(1).iloc[0]
    eye_hits = eye_hits_df[
        eye_hits_df.index.get_level_values('grasp_num') == current_grasp.grasp_num
    ].copy().reset_index()
    eye_hits['lookahead_grasp_num'] = (
        eye_hits
        .eye_hit
        .apply(lambda s: grasp_diff(s, grasp_df.head(-1)))
    )
    eye_hits['lookahead_distance'] = eye_hits.lookahead_grasp_num - eye_hits.grasp_num
    return eye_hits


def get_lookahead_grasps(grp):
    grp_cols = ['subject_id', 'trial_num', 'trial_type', 'grasp_num', 'current_grasp_object',
                'next_grasp_object']
    eye_hits_df, name = grp

    grasps_df = (
        eye_hits_df
        # this is to get all the grasps
        .groupby(grp_cols)
        .size()
        .to_frame('eye_hit_object_count')
        .reset_index()
        # this will make sure we implement the reverse expanding window
        .sort_values('grasp_num', ascending=False)
    )
    eye_hits_df = (
        grasps_df
        .pipe(expanding_pipe_df, partial(compute_lookahead_distance, eye_hits_df=eye_hits_df))
        .set_index(grp_cols)
    )
    return eye_hits_df



def expanding_pipe(df, window_function):
    return pd.Series([df.iloc[0:i].pipe(window_function)
                      for i in range(1, len(df) + 1)],
                     index=df.index)


def get_most_fixated(grp_df, n=0):
    grp_df = (
        grp_df
        .reset_index()
        .sort_values('hit_counts', ascending=False)
        .pipe(lambda df: df[~df.eye_hit.isin([
            'Other',
            df.current_grasp_object.values[0],
#             df.next_grasp_object.values[0],
        ])])
    )
    if len(grp_df) == 0:
        return np.nan
    elif len(grp_df) < n + 1:
        return np.nan
    else:
        return grp_df.eye_hit.iloc[n]


def compute_distance(df, col='most_fixated'):
    if len(df) <= 1:
        return np.nan
    fixated_object = df.tail(1)[col].values[0]
    if pd.isna(fixated_object):
        return np.nan
    grasped_df = (
        df
        .head(-1)
        .reset_index()
        .query('current_grasp_object == @fixated_object')
    )
    if len(grasped_df) == 0:
        return np.nan
    return (
        grasped_df.tail(1).grasp_num.values[0] - df.reset_index().tail(1).grasp_num.values[0]
    )


def get_grasp_distance(grp_df, col='most_fixated'):
    grp_df = grp_df.sort_values('grasp_num', ascending=False)
    grp_df['grasp_distance'] = (
        grp_df
        .pipe(expanding_pipe, partial(compute_distance, col=col))
    )
    grp_df = grp_df.sort_values('grasp_num', ascending=True)
    return grp_df


def grasp_distance(grp):
    grp_cols = ['subject_id', 'trial_num', 'trial_type']
    sample_df, name = grp
    sample_df = sample_df.sort_values(by='timestamp_dt')

    grasp_times = (
        sample_df.query('grasp_onset_bool == 1')
        [['pickup_location', 'timestamp_dt']]
        .rename(columns=dict(
            timestamp_dt='current_grasp_time',
            pickup_location='current_pickup_loc',
        ))
    )
    grasp_times['current_drop_loc'] = (
        sample_df
        .query('grasp_end_bool == 1')
        .drop_location
        .values
    )
    grasp_times['current_drop_time'] = (
        sample_df
        .query('grasp_end_bool == 1')
        .timestamp_dt
        .values
    )

    grasp_times = (
        grasp_times
        .sort_values(by='current_grasp_time')
        .reset_index(drop=True)
    )

    grasp_times['next_grasp_time'] = grasp_times.current_grasp_time.shift(-1)
    grasp_times['next_dropoff_time'] = grasp_times.current_drop_time.shift(-1)

    grasp_times['next_pickup_loc'] = grasp_times.current_pickup_loc.shift(-1)
    grasp_times['next_dropoff_loc'] = grasp_times.current_drop_loc.shift(-1)
    grasp_times['grasp_num'] = grasp_times.index.values

    distance_df = (
        grasp_times
        .head(-1)
        .pipe(
            lambda df:
                df.assign(
                    current_grasp_distance=(
                        df[['current_pickup_loc', 'current_drop_loc']]
                        .apply(
                            lambda row: shelf_distance(
                            row.current_pickup_loc,
                            row.current_drop_loc
                            ),
                            axis=1
                        )
                    ),

                    next_grasp_distance_from_pickup=(
                        df[['current_pickup_loc', 'next_pickup_loc']]
                        .apply(
                            lambda row: shelf_distance(
                             row.current_pickup_loc,
                             row.next_pickup_loc
                             ),
                            axis=1
                        )
                    ),
                    next_grasp_distance_from_dropoff=(
                        df[['current_drop_loc', 'next_pickup_loc']]
                        .apply(
                            lambda row: shelf_distance(
                            row.current_drop_loc,
                            row.next_pickup_loc
                            ),
                            axis=1
                        )
                    ),
                )
                .pipe(lambda df:
                    df.assign(
                        grasp_interval_from_onset=(
                        df.next_grasp_time
                        - df.current_grasp_time) / np.timedelta64(1, 's'),
                        grasp_interval_off_on=(
                        df.next_grasp_time -
                        df.current_drop_time) / np.timedelta64(1, 's'),
                        subject_id=name[0],
                        trial_num=name[1],
                        trial_type=name[2]
                    )
                )
        )
        .set_index(grp_cols)
    )

    return distance_df
