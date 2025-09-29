import numpy as np
import pandas as pd


def save_tracks_3D_to_csv_and_return_dataFrame(tracks_3D, savepath):
    ''' Save the trajectories to a csv file. This only works for data
        of the form (numFrames, numFish=2, numBodyPoints=3, numDims=3),
        because we title each column. Saved to 2 decimal places of precision.

    --- args ---
    tracks_3D: tracking data, shape
              (numFrames, numFish=2, numBodyPoints=3, numDims=3)
    savepath: string, location to save to.
    '''
    numFrames, numFish, numBodyPoints, _ = tracks_3D.shape
    column_headings = ['fish1_head_x', 'fish1_head_y', 'fish1_head_z',
                       'fish1_pec_x', 'fish1_pec_y', 'fish1_pec_z',
                       'fish1_tail_x', 'fish1_tail_y', 'fish1_tail_z',
                       'fish2_head_x', 'fish2_head_y', 'fish2_head_z',
                       'fish2_pec_x', 'fish2_pec_y', 'fish2_pec_z',
                       'fish2_tail_x', 'fish2_tail_y', 'fish2_tail_z']

    column_data_list = []
    for fishIdx in range(numFish):
        for bpIdx in range(numBodyPoints):
            for dimIdx in range(3):
                col = np.copy(tracks_3D[:, fishIdx, bpIdx, dimIdx])
                column_data_list.append(col)
    column_data = np.stack(column_data_list, axis=1)

    tracks_3D_df = pd.DataFrame(data=column_data, columns=column_headings)

    tracks_3D_df.to_csv(savepath, index_label='frame_index', sep=',', mode='w', float_format='%.2f')

    return tracks_3D_df