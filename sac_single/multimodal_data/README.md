
## Multimodal data for BC (written by jhpark)

DIMENSION : (episode * horrizon * dim) # 1000 episode, 1000 horrizon

1. data_obs.npy # observation
dim : 2(robot) + 2(red_block) + 2(blue_block) = 6

2. data_act.npy # action
dim : 2(delta x, delta y)

3. data_msk.npy # mask for valid timestep
dim : 1(valid 1, otherwise 0)
