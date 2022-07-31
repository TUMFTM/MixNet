## Parameter description
All parameters can be set in [main_params](main_params.ini).

### Section : `MODEL_PARAMS`

| Parameter | Type | Default | Description
| ------------- | ------------- | ------ | ----- |
model_type | str | MixNet | Choose prediction model, options: MixNet, IndyNet|
sampling_frequency | float | 10.0 | Frequency in Hz to sample object history (network input) |
data_min_obs_length | float | 1.1 | Minimal observation length to output a prediction |
data_min_velocity | float | 0.0 | Velocity threshold in m/s, above data-based prediction is used |
data_max_acceleration | float | 0.0 | Acceleration threshold in m/s², below data-based prediction is used |
stat_vel_threshold | float | 5.0 | Distance of minimal movement in m, below object is predicted static |
stat_prediction_horizon | float | 5.0 | Static prediction horizon in s |
view | int | 400 | Length in m to sample boundaries in driving direction (network input) |
dist | int | 20 | Sample distance in m of boundaries in driving direction (network input) |

### Section : `FILE_NAMES`
Specify the file names for MixNet, IndyNet and map, which should be used by the module. All files have to be stored in the respective subfolders in `mix_net/mix_net/data`.


### Section : `MIX_NET_PARAMS`
| Parameter | Type | Default | Description
| ------------- | ------------- | ------ | ----- |
physics_based_const_vel | boolean | false | If true, constant velocity assumption is applied |
physics_based_init_vel | boolean| true | If true, intial velocity from observation storage is applied |
safety_physics_override | boolean | true | If true, lateral offset at the beginning of the prediction horizon is correct |
override_error | float | 2.0 | Distance threshold in m to apply override function |
pred_len | int | 50 | Number of prediction steps
dt | float | 0.1 | Time step size in s between prediction steps |
data_min_obs_length | int | 5 | Minimal number of observation to apply data-based prediction


### Section : `INTERACTION_PARAMS`
| Parameter | Type | Default | Description
| ------------- | ------------- | ------ | ----- |
rule_based | boolean | true | If true, interaction between the objects by fuzzy logic is considered |
no_iterations | int | 1 | Number of iterations the fuzzy logic is applied to the objects to resolve conflicts |
priority_on_ego | boolean | true | If true, collision free ego prediction is secured
delta_v_overtake | float | 5.0 | Velocity difference in m/s, above overtaking is modeled |
lanechange_time | float | 2.0 | Time in s of modeld overtaking maneuver |
lat_overtake_dist | float | 4.0 | Lateral distance in m of modeled overtaking maneuver |
lat_veh_half_m | float | 0.943 | Half vehicle width in m |
long_veh_half_m | float | 2.4605| Half vehicle length in m |
collision_check | str | euclidean | Type of collision check (if not 'euclidean' rectangle chek is applied) |
approx_radius | float | 2.0 | Radius in m used for euclidean collision check |
lat_safety_m | float | 1.5 | Lateral safety distance in m for collision check |
long_safety_m | float | 1.5 | Longitudinal safety distance in m for collision check |
