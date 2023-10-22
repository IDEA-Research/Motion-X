import os

path = '/comp_robot/linjing/data/Motion_Generation/AMASS/SMPLX_G/amass_data'
name_table = {
    'BMLrub': 'BioMotionLab_NTroje',
    'DFaust': 'DFaust_67',
    'HDM05': 'MPI_HDM05',
    'MoSh': 'MPI_mosh',
    'PosePrior': 'MPI_Limits',
    'SSM': 'SSM_synced',
    'TCDHands': 'TCD_handMocap',
    'Transitions': 'Transitions_mocap',
}
for item in os.listdir(path):
    if item in name_table:
        old_path = os.path.join(path, item)
        new_path = os.path.join(path, name_table[item])
        os.rename(old_path, new_path)