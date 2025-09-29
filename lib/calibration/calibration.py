import os

import numpy as np
from joblib import load


class Calibration:

    def __init__(self, calibration_folder_path):
        """ Instantiate the object

        -- args --
        calibration_folder_path: the path to a calibration folder, where the regressed functions
                                 have already been computed and saved

        """
        # record the folder paths
        self.calibration_folder_path = calibration_folder_path
        self.python_calibration_folderPath = os.path.join(self.calibration_folder_path,
                                                          'python_calibration_models')

        # load the models and assign as attributes
        self._load_models()

    def _load_models(self):
        """ Instantiate the regression object attributes:
            xyz_getter, imCoord_getter, xz_getter, xy_getter, yz_getter
        """
        imCoords_to_XYZ_path = os.path.join(self.python_calibration_folderPath, 'imCoords_to_XYZ.joblib')
        xy_yz_to_xz_path = os.path.join(self.python_calibration_folderPath, 'xy_yz_to_xz.joblib')
        xz_yz_to_xy_path = os.path.join(self.python_calibration_folderPath, 'xz_yz_to_xy.joblib')
        xz_xy_to_yz_path = os.path.join(self.python_calibration_folderPath, 'xz_xy_to_yz.joblib')

        self.xyz_getter = load(imCoords_to_XYZ_path)
        self.xz_getter = load(xy_yz_to_xz_path)
        self.xy_getter = load(xz_yz_to_xy_path)
        self.yz_getter = load(xz_xy_to_yz_path)
        return

        # ---- Main Methods ---- #

    def compute_imageCoord_triplet_from_XYZ(self, XYZ):
        """ Predict the image coordinates in all 3 camera views of the
            3D point XYZ

        -- inputs --
        XYZ: array (3,), the position of a point in 3D

        -- returns --
        imCoords: array (3,2) of image coordinates in standard camera
                  order of XZ,XY,YZ
        """
        imCoords = self.imCoord_getter.predict(XYZ.reshape(1, -1))
        imCoords = imCoords.reshape(3, 2)
        return imCoords

    def compute_XYZ_from_imageCoord_triplet(self, imCoords):
        ''' Predict the XYZ position of the point given by the image
            coordinates from all 3 cameras

        -- Inputs --
        imCoords: array of shape (3,2)

        -- Outputs --
        XYZ: array of shape (3)

        '''
        XYZ = self.xyz_getter.predict(imCoords.reshape(-1, 6))
        return XYZ

    def compute_XZ_imcoords_from_XY_YZ(self, xy_imCoord, yz_imCoord):
        ''' Given an image coordinate from both the XY and YZ views,
            compute the corresponding image coordinate from the XZ view

        -- args --
        xy_imCoord: image coordinate of shape (2,)
        yz_imCoord: image coordinate of shape (2,)

        -- returns --
        xz_imCoord: image coordinate of shape (2,)

        '''
        input_data = np.hstack((xy_imCoord, yz_imCoord)).reshape(1, 4)
        xz_imCoord = self.xz_getter.predict(input_data)
        return xz_imCoord

    def compute_XY_imcoords_from_XZ_YZ(self, xz_imCoord, yz_imCoord):
        ''' Given an image coordinate from both the XZ and YZ views,
            compute the corresponding image coordinate from the XY view

        -- args --
        xz_imCoord: image coordinate of shape (2,)
        yz_imCoord: image coordinate of shape (2,)

        -- returns --
        xy_imCoord: image coordinate of shape (2,)
        '''
        # prepare the input for predictor, and predict the imcoord
        input_data = np.hstack((xz_imCoord, yz_imCoord)).reshape(1, 4)
        xy_imCoord = self.xy_getter.predict(input_data)
        return xy_imCoord

    def compute_YZ_imcoords_from_XZ_XY(self, xz_imCoord, xy_imCoord):
        ''' Given an image coordinate from both the XY and YZ views,
            compute the corresponding image coordinate from the XZ view

        -- args --
        xz_imCoord: image coordinate of shape (2,)
        xy_imCoord: image coordinate of shape (2,)

        -- returns --
        yz_imCoord: image coordinate of shape (2,)

        '''
        # prepare the input for predictor, and predict the imcoord
        input_data = np.hstack((xz_imCoord, xy_imCoord)).reshape(1, 4)
        yz_imCoord = self.yz_getter.predict(input_data)
        return yz_imCoord


    def compute_point_correspondence_error(self, cam_ids, image_coordinates_cam1, image_coordinates_cam2):

        """ computes the point correspondence error of the coordinates given the coordinates from two cameras.

            -- args ---
             camIdxs: a list denoting the cameras the imCoords args are coming from.
                        Has to be [0,1], [1,2], or [0, 2]
            imCoords_cam1: image coordinates from a camera (x, y)
            imCoords_cam2: image coordinates from a different camera (x, y)
           """

        # The error is NaN if either point is NaN
        if np.all(np.isnan(image_coordinates_cam1)) or np.all(np.isnan(image_coordinates_cam2)):
            return np.NaN

        proposed_im_coords = self.get_proposed_coordinates_triplet_from_two_cameras(cam_ids, image_coordinates_cam1,
                                                                                    image_coordinates_cam2)
        error = self.compute_error_of_proposed_coordinates(proposed_im_coords)
        return error

    def get_proposed_coordinates_triplet_from_two_cameras(self, cam_ids, image_coordinates_cam1,
                                                          image_coordinates_cam2):

        """ Uses information from two cameras to create the coordinates for the third

         -- args ---
          camIdxs: a list denoting the cameras the imCoords args are coming from.
                     Has to be [0,1], [1,2], or [0, 2]
         imCoords_cam1: image coordinates from a camera
         imCoords_cam2: image coordinates from a different camera
        """

        if cam_ids == [0, 1]:
            # derive YZ
            imCoords_cam3 = self.compute_YZ_imcoords_from_XZ_XY(image_coordinates_cam1, image_coordinates_cam2)
            proposed_im_coords = np.vstack((image_coordinates_cam1, image_coordinates_cam2, imCoords_cam3))
        elif cam_ids == [0, 2]:
            # derive XY
            imCoords_cam3 = self.compute_XY_imcoords_from_XZ_YZ(image_coordinates_cam1, image_coordinates_cam2)
            proposed_im_coords = np.vstack((image_coordinates_cam1, imCoords_cam3, image_coordinates_cam2))
        elif cam_ids == [1, 2]:
            # derive XZ
            imCoords_cam3 = self.compute_XZ_imcoords_from_XY_YZ(image_coordinates_cam1, image_coordinates_cam2)
            proposed_im_coords = np.vstack((imCoords_cam3, image_coordinates_cam1, image_coordinates_cam2))
        else:
            raise (ValueError, "cam_ids need to be one of [0,1], [1,2], [0,2]")
        return proposed_im_coords

    def compute_error_of_proposed_coordinates(self, proposed_image_coordinates):

        """
        For each pairing of cameras, compute the 3rd cam image coordinate,
        then compare this triplet to the proposed_image_coordinates, which act as truth
        Note1: If this is a good pairing, then proposed_image_coordinates represent the same point in 3D
        Note2: for one of these camera pairings test, we will get back an error of 0,
                since we did the same computation to compute proposed_coordinates.
        """

        xz_coordinates = proposed_image_coordinates[0]
        xy_coordinates = proposed_image_coordinates[1]
        yz_coordinates = proposed_image_coordinates[2]

        derived_xz = self.compute_XZ_imcoords_from_XY_YZ(xy_coordinates, yz_coordinates)
        image_coords_derXZ = np.vstack((derived_xz, xy_coordinates, yz_coordinates))
        error_derXZ = np.linalg.norm(proposed_image_coordinates - image_coords_derXZ)

        derived_xy = self.compute_XY_imcoords_from_XZ_YZ(xz_coordinates, yz_coordinates)
        image_coords_derXY = np.vstack((xz_coordinates, derived_xy, yz_coordinates))
        error_derXY = np.linalg.norm(proposed_image_coordinates - image_coords_derXY)

        derived_yz = self.compute_YZ_imcoords_from_XZ_XY(xz_coordinates, xy_coordinates)
        image_coords_derYZ = np.vstack((xz_coordinates, xy_coordinates, derived_yz))
        error_derYZ = np.linalg.norm(proposed_image_coordinates - image_coords_derYZ)

        errors = np.vstack((error_derXZ, error_derXY, error_derYZ))
        error = np.sum(errors)

        return error
