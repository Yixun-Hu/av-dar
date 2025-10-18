import os
import numpy as np
import scipy.io.wavfile

HRIR_dataset_dir = '<path-to-hearing-anything-anywhere>/HRIRs'
hrir_cache = {}
# borrowed from https://github.com/maswang32/hearinganythinganywhere/
def get_HRIR(azimuth, elevation):
    """
    Returns 2-channel (2x256) HRIR given an azimuth and elevation angle in degrees.
    Azimuth angles are anti-clockwise.
    """
    negative = elevation < 0
    elevation = abs(elevation)
    
    if negative:
        if elevation <= 7.5:
            suffix = "0,0.wav"
        elif 7.5 <= elevation <  16.25:
            suffix = "-15,0.wav"
        elif 16.25 <= elevation < 21.25:
            suffix = "-17,5.wav"
        elif 21.25 <= elevation < 27.5:
            suffix = "-25,0.wav"
        elif 27.5 <= elevation < 32.65:
            suffix = "-30,0.wav"
        elif 32.65 <= elevation < 40.15:
            suffix = "-35,3.wav"
        elif 40.15 <= elevation < 49.5:
            suffix = "-45,0.wav"
        elif 49.5 <= elevation < 57:
            suffix = "-54,0.wav"
        elif 57 <= elevation < 62.4:
            suffix = "-60,0.wav"
        elif 62.4 <= elevation < 69.9:
            suffix = "-64,8.wav"        
        elif 69.9 <= elevation < 78:
            suffix = "-75,0.wav"
        elif elevation >= 78:
            suffix = "-81,0.wav"
    else:
        if elevation <= 7.5:
            suffix = "0,0.wav"
        elif 7.5 <= elevation <  16.25:
            suffix = "15,0.wav"
        elif 16.25 <= elevation < 21.25:
            suffix = "17,5.wav"
        elif 21.25 <= elevation < 27.5:
            suffix = "25,0.wav"
        elif 27.5 <= elevation < 32.65:
            suffix = "30,0.wav"
        elif 32.65 <= elevation < 40.15:
            suffix = "35,3.wav"
        elif 40.15 <= elevation < 49.5:
            suffix = "45,0.wav"
        elif 49.5 <= elevation < 57:
            suffix = "54,0.wav"
        elif 57 <= elevation < 62.4:
            suffix = "60,0.wav"
        elif 62.4 <= elevation < 69.9:
            suffix = "64,8.wav"        
        elif 69.9 <= elevation < 82.5:
            suffix = "75,0.wav"
        elif elevation >= 82.5:
            suffix = "90,0.wav"

    azimuth = str(int(np.round(azimuth) % 360))
    path = os.path.join(HRIR_dataset_dir, "azi_" + azimuth + ",0_ele_" + suffix)

    global hrir_cache
    if path in hrir_cache:
        return hrir_cache[path]
    
    _, hrir = scipy.io.wavfile.read(path)

    # 32 bit - convert to float
    hrir = hrir.T/2147483648

    hrir_cache[path] = hrir
    return hrir

def compute_hrirs(incoming_listener_directions, listener_forward, listener_left):
    """
    Returns a stack of head-related IRs for each listener direction.

    Parameters
    ----------
    incoming_listener_directions: (P,3). Points towards the listener.
    listener_forward: (3,). Vector pointing in the forward direction of listener.
    listener_left: (3,). Vector pointing to the left of the listener.

    Returns
    -------
    (P,2,256) np.array of HRIRs
    """
    norms = np.linalg.norm(incoming_listener_directions, axis=-1).reshape(-1,1)
    incoming_listener_directions = -incoming_listener_directions/norms
    listener_forward = listener_forward/np.linalg.norm(listener_forward)

    # import IPython; IPython.embed();
    #listener_left points left (right-handed coordinate system with listener_forward)
    listener_left = listener_left/np.linalg.norm(listener_left)

    #Make sure listener_forward and listener_left are orthogonal
    assert np.abs(np.dot(listener_forward, listener_left)) < 0.01

    listener_up = np.cross(listener_forward, listener_left)
    head_basis = np.stack((listener_forward, listener_left, listener_up), axis=-1)

    #Compute Azimuths and Elevation
    head_coordinates = incoming_listener_directions @ head_basis
    azimuths = np.degrees(np.arctan2(head_coordinates[:, 1], head_coordinates[:, 0]))
    elevations = np.degrees(np.arctan(head_coordinates[:, 2]/np.linalg.norm(head_coordinates[:, 0:2],axis=-1)+1e-8))

    #Retrieve HRIRs
    h_rirs = np.zeros((incoming_listener_directions.shape[0], 2, 256))
    for i in range(incoming_listener_directions.shape[0]):
        h_rirs[i] = get_HRIR(azimuth=azimuths[i], elevation=elevations[i])

    return h_rirs

