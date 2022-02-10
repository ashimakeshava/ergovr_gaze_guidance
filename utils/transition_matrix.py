import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


def plot_net_transition_per_trial(tm_df, PLOT_PATH, plot=False):

    os.makedirs(os.path.dirname(PLOT_PATH), exist_ok=True)

    sym_dict = {'trial_num': [], 'grasp_num': [], 'sym_index': [], 'trial_type': [], 'entropy':[], }

    for trial_num in tm_df.trial_num.unique():

        tmpdf = (
            tm_df
            .query('trial_num == @trial_num')
            .query('is_fixation == True ')
            .drop_duplicates(subset=['is_fixation', 'fix_duration', 'fix_type', 'eye_hit'], keep='first')
            [['trial_num', 'fix_type', 'grasp_num', 'trial_type']]
        )

        trial_type = tmpdf.trial_type.unique()[0]
#         epoch_time
        grasp_num = tmpdf.grasp_num.max()
        tmpdf.dropna(subset=['fix_type'], inplace=True)

        tmpdf['fix_type_destination'] = tmpdf.groupby(['grasp_num']).fix_type.shift(-1)

        tmpdf = (
            tmpdf
            .groupby(['fix_type', 'fix_type_destination'])
            .size()
            .rename('num_switch')
            .reset_index()
        )
        tmpdf.loc[tmpdf.fix_type == tmpdf.fix_type_destination, 'num_switch'] = 0

        tmpdf = tmpdf.pivot(index='fix_type', columns = 'fix_type_destination', values='num_switch')

        columns = ['prev_TO', 'prev_TS', 'current_TO', 'current_TS', 'next_TO', 'next_TS', 'other']
#         print(tmpdf.columns, tmpdf.index.values)
        for col in columns:
            if col not in tmpdf.index.values or col not in tmpdf.columns:
                tmpdf.loc[col, col] = 0

#         print(tmpdf.columns, tmpdf.index.values)
        tmpdf = tmpdf.reindex(
            [
                'prev_TO', 'prev_TS',
                'current_TO', 'current_TS',
                'next_TO','next_TS',
                'other'
            ]
        )

        tmpdf = tmpdf.loc[:, [
                'prev_TO', 'prev_TS',
                'current_TO', 'current_TS',
                'next_TO','next_TS',
                'other'
        ]]

#         print(tmpdf)
#         if normalize:
#             tmpdf = tmpdf.div(tmpdf.sum(axis=1), axis=0)
# #             tmpdf = tmpdf.div(tmpdf.sum(axis=1).sum())

#
        tmpdf = tmpdf.fillna(0)

        A_net = (tmpdf - tmpdf.T)
        A_total = (tmpdf + tmpdf.T)

        F = (np.tril(abs(A_net.to_numpy())).sum())/(np.tril(A_total.to_numpy()).sum())

        Px = tmpdf.div(tmpdf.sum(axis=1), axis=0)
        Px = Px.fillna(0).to_numpy()
        logPx = np.log2(Px)
        logPx[logPx == -np.inf] = np.nan
        logPx[logPx == np.inf] = np.nan
        entropy = -np.nansum((np.nansum((Px*logPx), axis=1)), axis=0)

#         print(Px, np.log2(Px))

        sym_dict['trial_num'].append(trial_num)
        sym_dict['sym_index'].append(F)
        sym_dict['trial_type'].append(trial_type)
        sym_dict['grasp_num'].append(grasp_num)
        sym_dict['entropy'].append(entropy)

#     return sym_dict

        if plot:
            sns.set(context = "poster", style="white", palette="muted", font_scale=1, rc={'figure.figsize':(30,10)})
            fig, ax = plt.subplots(1, 3, gridspec_kw={'width_ratios':[1, 1, 1,]})
            ax = ax.ravel()
            mask = np.zeros_like(A_net.to_numpy())
            mask[np.triu_indices_from(mask)] = True

            g0 = sns.heatmap(
                tmpdf,
                cmap='Blues',
                square=True, annot=True, cbar=True,
                vmin=0, vmax=10,
                linewidths=.5, ax=ax[0]
            )
            g0.set(xlabel='', ylabel='origin');

            g1 = sns.heatmap(
                A_net,
                cmap='coolwarm', center=0,
                square=True, annot=True, cbar=True,
                vmin=-10, vmax=10, mask=mask,
                linewidths=.5, ax=ax[1]
            )

            g1.set(
                xlabel='destination', ylabel='', yticks=[]
    #             title=f'Trial Num: {trial_num}, Trial Type: {trial_type}, nr. grasp: {grasp_num} '
            );

            g2 = sns.heatmap(
                A_total,
                cmap='Blues',
                square=True, annot=True, cbar=True,
                vmin=0, vmax=10, mask=mask,
                linewidths=.5, ax=ax[2]
            )
            g2.set(xlabel='', ylabel='', yticks=[]);



            plt.yticks(rotation=0, fontsize=25)
            plt.xticks(rotation=90, fontsize=25)

            plt.suptitle(
                f'Trial Num: {trial_num}, Trial Type: {trial_type}, nr. grasp: {grasp_num}, F: {F*100:.2f}%')
            plt.savefig(
                f'{PLOT_PATH}/transition_matrix_trial_{int(trial_num)}.png',
                quality=90,
                bbox_inches='tight')
            plt.close()

    return sym_dict


def plot_net_transition_per_trial_per_grasp(tm_df, PLOT_PATH, plot=True):
    os.makedirs(os.path.dirname(PLOT_PATH), exist_ok=True)

    sym_dict = {'trial_num': [], 'grasp_num': [], 'sym_index': [], 'trial_type': [], 'entropy':[],}

    transitions_df = pd.DataFrame()

    for trial_num in tm_df.trial_num.unique():
        for grasp_num in tm_df.query('trial_num == @trial_num').grasp_num.unique():

            tmpdf = (
                tm_df
                .query('grasp_num == @grasp_num and trial_num == @trial_num')
                .query('is_fixation == True ')
                .drop_duplicates(subset=['is_fixation', 'fix_duration', 'fix_type', 'eye_hit'], keep='first')
                [['trial_num', 'fix_type', 'grasp_num', 'trial_type',]]
            )
            if tmpdf.shape[0]==0:
                continue

            trial_type = tmpdf.trial_type.unique()[0]
    #         grasp_num = tmpdf.grasp_num.max()
            tmpdf.dropna(subset=['fix_type'], inplace=True)

            tmpdf['fix_type_destination'] = tmpdf.groupby(['grasp_num']).fix_type.shift(-1)

            tmpdf = (
                tmpdf
                .groupby(['fix_type', 'fix_type_destination'])
                .size()
                .rename('num_switch')
                .reset_index()
            )
            tmpdf.loc[tmpdf.fix_type == tmpdf.fix_type_destination, 'num_switch'] = 0

            tmpdf = tmpdf.pivot(index='fix_type', columns = 'fix_type_destination', values='num_switch')

            columns = ['prev_TO', 'prev_TS', 'current_TO', 'current_TS', 'next_TO', 'next_TS', 'other']
    #         print(tmpdf.columns, tmpdf.index.values)
            for col in columns:
                if col not in tmpdf.index.values or col not in tmpdf.columns:
                    tmpdf.loc[col, col] = 0

    #         print(tmpdf.columns, tmpdf.index.values)
            tmpdf = tmpdf.reindex(
                [
                    'prev_TO', 'prev_TS',
                    'current_TO', 'current_TS',
                    'next_TO','next_TS',
                    'other'
                ]
            )

            tmpdf = tmpdf.loc[:, [
                    'prev_TO', 'prev_TS',
                    'current_TO', 'current_TS',
                    'next_TO','next_TS',
                    'other'
            ]]

            tmpdf = tmpdf.fillna(0)

            t_matrix_df = tmpdf.stack().reset_index()
            t_matrix_df.columns = ['origin', 'destination', 'num_switch']

            A_net = (tmpdf - tmpdf.T)
            A_total = (tmpdf + tmpdf.T)

            F = (np.tril(abs(A_net.to_numpy())).sum())/(np.tril(A_total.to_numpy()).sum())

            net_df = A_net.where(np.tril(np.ones(A_net.shape)).astype(np.bool))
            net_df = net_df.stack().reset_index()
            net_df.columns = ['origin', 'destination', 'net_transition']

            #print(net_df.shape)

            ### swap direction of transition if net transition is negative

            total_df = A_total.where(np.tril(np.ones(A_total.shape)).astype(np.bool))
            total_df = total_df.stack().reset_index()
            total_df.columns = ['origin', 'destination', 'total_transition']

            df = pd.merge(net_df, total_df, on=['origin', 'destination'], how='inner')

            df['num_switch'] = 0.5* (df.net_transition + df.total_transition)

            df.loc[df.net_transition<0, ['origin', 'destination']] = (
                df.loc[df.net_transition<0, ['destination', 'origin']].values
            )
            # df = pd.merge(t_matrix_df, df, on=['origin', 'destination'], how='inner')

            df['trial_num'] = trial_num
            df['trial_type'] = trial_type
            df['grasp_num'] = grasp_num

            df = df.query('origin != destination')

            df.net_transition = df.net_transition.abs()

            df['ori_net_transition'] = df.groupby('origin').net_transition.transform('sum')

            # df['ori_total_transition'] = df.groupby('origin').num_switch.transform('sum')
            #
            df['rel_transition'] = df.net_transition/df.ori_net_transition
            #
            # df['prob_num_switch'] = df.num_switch/df.ori_total_transition
            #
            df.loc[df.rel_transition == np.inf, 'rel_transition'] = 0
            #
            # df.loc[df.prob_num_switch == np.inf, 'prob_num_switch'] = 0

            transitions_df = pd.concat([transitions_df, df], ignore_index=True)

            if df.shape[0] != 21: print(df.trial_num.unique(), tm_df.subject_id.unique())

            # Px = tmpdf.div(tmpdf.sum(axis=1), axis=0)
            # Px = Px.fillna(0).to_numpy()
            # logPx = np.log2(Px)
            # logPx[logPx == -np.inf] = np.nan
            # logPx[logPx == np.inf] = np.nan
            # entropy = -np.nansum((np.nansum((Px*logPx), axis=1)), axis=0)

    #         print(Px, np.log2(Px))

            sym_dict['trial_num'].append(trial_num)
            sym_dict['sym_index'].append(F)
            sym_dict['trial_type'].append(trial_type)
            sym_dict['grasp_num'].append(grasp_num)
            sym_dict['entropy'].append(0)

            if plot:
                sns.set(context = "poster", style="white", palette="muted", font_scale=1, rc={'figure.figsize':(30,10)})
                fig, ax = plt.subplots(1, 3, gridspec_kw={'width_ratios':[1, 1, 1,]})
                ax = ax.ravel()
                mask = np.zeros_like(A_net.to_numpy())
                mask[np.triu_indices_from(mask)] = True

                g0 = sns.heatmap(
                    tmpdf,
                    cmap='Blues',
                    square=True, annot=True, cbar=True,
                    vmin=0, vmax=5,
                    linewidths=.5, ax=ax[0]
                )
                g0.set(xlabel='', ylabel='origin');

                g1 = sns.heatmap(
                    A_net,
                    cmap='coolwarm', center=0,
                    square=True, annot=True, cbar=True,
                    vmin=-2, vmax=2, mask=mask,
                    linewidths=.5, ax=ax[1]
                )

                g1.set(
                    xlabel='destination', ylabel='', yticks=[]
        #             title=f'Trial Num: {trial_num}, Trial Type: {trial_type}, nr. grasp: {grasp_num} '
                );

                g2 = sns.heatmap(
                    A_total,
                    cmap='Blues',
                    square=True, annot=True, cbar=True,
                    vmin=0, vmax=5, mask=mask,
                    linewidths=.5, ax=ax[2]
                )
                g2.set(xlabel='', ylabel='', yticks=[]);



                plt.yticks(rotation=0, fontsize=25)
                plt.xticks(rotation=90, fontsize=25)

                plt.suptitle(
                    f'Trial Num: {trial_num}, Trial Type: {trial_type}, nr. grasp: {grasp_num}, F: {F*100:.2f}%')
                plt.savefig(
                    f'{PLOT_PATH}/transition_matrix_trial_{int(trial_num)}_{int((grasp_num))}.pdf',
                    quality=90,
                    bbox_inches='tight')
                plt.close()

    return sym_dict, transitions_df
