import numpy as np
import math

class Doggy_robot():
    def __init__(self,sim,robot_name,target_name):
        self.sim = sim
        self.name = robot_name
        self.target_name = target_name
        self.joint_list = [
            "RR_upper_leg_joint", "FL_upper_leg_joint",
            "FR_upper_leg_joint", "RL_upper_leg_joint",
            "RR_lower_leg_joint", "FL_lower_leg_joint",
            "FR_lower_leg_joint", "RL_lower_leg_joint", 
        ]
        self.links_list = ["base_link_visual",
            "RR_upper_leg_link_visual","RR_lower_leg_link_visual","RR_foot_link_visual",
            "RL_upper_leg_link_visual","RL_lower_leg_link_visual","RL_foot_link_visual",
            "FR_upper_leg_link_visual","FR_lower_leg_link_visual","FR_foot_link_visual",
            "FL_upper_leg_link_visual","FL_lower_leg_link_visual","FL_foot_link_visual",
            ]
        
        self.positions = ["RL","RR","FL","FR"]
        self.robot = self.sim.getObject(f'/{self.name}/')
        self.target = self.sim.getObject(f'/{self.target_name}')
        self.fill_robot_data()
        self.last_x = self.sim.getObjectPosition(self.robot, self.target)[0]
        self.last_joints_postion = self.get_joints_position()
        self.last_joints_orientation = self.get_joints_orientation()
        self.last_time = self.sim.getSimulationTime()
        self.last_joints_speed = self.get_joints_speed()
        self.update_all_info()



    def update_all_info(self):
        self.time = self.sim.getSimulationTime()
        self.x,self.y,self.z = self.get_relative_position()
        self.delta_x = self.x - self.last_x
        self.roll,self.pitch,self.yaw = self.get_orientation()
        self.inverse_rotation_matrix = self.get_inverse_rotation_matrix()
        self.correct_dir_angle = self.get_correct_direction_angle()
        [self.vx, self.vy, self.vz],\
        [self.wx, self.wy, self.wz] = self.get_velocities()
        self.fall = self.check_fall()
        self.arrived = self.check_arrival()
        self.upside_down = self.inverse_rotation_matrix[2][2] < 0
    

        self.joints_position = []
        self.joints_speed = []
        self.joints_onMax = []
        self.joints_orientation =[]
        self.joints_0_speed = []
        for jointName in self.joint_list:
            joint = self.handleDict[jointName]
            angle = self.sim.getJointPosition(joint)
            _, speed = self.sim.getObjectFloatParameter(joint, self.sim.jointfloatparam_velocity)
            self.joints_position.append(angle)
            self.joints_speed.append(speed)
            self.joints_onMax.append(self.check_joint_on_max(jointName,angle))
            self.joints_orientation.append(1) if speed > 0 else self.joints_orientation.append(-1)
            self.joints_0_speed.append(self.get_joint_speed_0(speed))


        self.joints_accel = self.get_joints_acceleration()
        self.n_orientation_changes = np.sum(np.array(self.joints_orientation) != np.array(self.last_joints_orientation))
        self.cg_inertial_orientation = self.get_cg()
        self.cg_base_link_orientation = self.inverse_rotation_matrix @ self.cg_inertial_orientation

        self.elbows_matrix = {}
        self.foots_matrix = {}
        self.inertial_coord_foots_matrix = {}
        self.n_elbows_above_feet = 0
        for position in self.positions:
            self.foots_matrix[position] = self.get_joint_final_matrix(position,False)
            self.elbows_matrix[position] = (self.get_elbow_final_matrix(position,False))
            self.inertial_coord_foots_matrix[position] = self.inverse_rotation_matrix @ self.foots_matrix[position]
            if self.foots_matrix[position][2][3] < self.elbows_matrix[position][2][3]:
                self.n_elbows_above_feet += 1

        self.base_point_polygons = {"RL":(self.foots_matrix["RL"] [0][3],self.foots_matrix["RL"] [1][3]),
                "RR":(self.inertial_coord_foots_matrix["RR"] [0][3],self.inertial_coord_foots_matrix["RR"] [1][3]),
                "FL":(self.inertial_coord_foots_matrix["FL"] [0][3],self.inertial_coord_foots_matrix["FL"] [1][3]),
                "FR":(self.inertial_coord_foots_matrix["FR"] [0][3],self.inertial_coord_foots_matrix["FR"] [1][3])}

        self.poligon_area = self.get_poligon_area()
        self.cg_inside = self.check_cg_inside()
        self.n_feet_above_base_link = self.get_feet_above_base_link()
        self. crossing_foot = [self.foots_matrix["RR"][0][3] > self.foots_matrix["FR"][0][3],self.foots_matrix["RL"][0][3] > self.foots_matrix["FL"][0][3] ]
        
        r_distance = math.sqrt((self.foots_matrix["FR"][0][3]-self.foots_matrix["RR"][0][3])**2+(self.foots_matrix["FR"][1][3]-self.foots_matrix["RR"][1][3])**2+(self.foots_matrix["FR"][2][3]-self.foots_matrix["RR"][2][3])**2)
        l_distance = math.sqrt((self.foots_matrix["FL"][0][3]-self.foots_matrix["RL"][0][3])**2+(self.foots_matrix["FL"][1][3]-self.foots_matrix["RL"][1][3])**2+(self.foots_matrix["FL"][2][3]-self.foots_matrix["RL"][2][3])**2)
        self.foot_distance = [r_distance,l_distance]

        self.last_time = self.time
        self.last_x = self.x
        self.last_joints_speed = self.joints_speed
        self.last_joints_orientation = self.joints_orientation



    def get_joints_orientation(self):
        speeds = np.array(self.get_joints_speed())
        orientation = np.where(speeds > 0, 1, np.where(speeds < 0, -1, 0))
        return orientation

    def check_joint_on_max(self,jointName,angle):
        if "upper" in jointName:
            if angle < -1.5 or angle > 1.5:
                return 1
            else:
                return 0
        elif "lower" in jointName:
            if angle > -0.1 or angle < -2.3:
                return 1
            else:
                return 0
        else:
            raise ValueError("Não foram observadas todas as juntas")

    def get_joints_acceleration(self):
        return (np.array(self.joints_speed)-np.array(self.last_joints_speed))/(self.time - self.last_time)

    def fill_robot_data(self):
        handleDict = {}
        for name in self.joint_list:
            handleDict[name] = self.sim.getObject(f"/{self.name}/{name}")
        self.handleDict = handleDict


        self.vertex_2_center = {}
        for vertex in ["RL","RR","FL","FR"]:
            base =  self.sim.getObject(f'/{self.name}/base_link_visual')
            upper_joint = self.sim.getObject(f'/{self.name}/{vertex}_upper_leg_joint')
            vertex_2_center = self.sim.getObjectPosition(upper_joint,base)
            self.vertex_2_center[vertex] = {}
            self.vertex_2_center[vertex]["x"] = vertex_2_center[0]
            self.vertex_2_center[vertex]["y"] = vertex_2_center[1]
            self.vertex_2_center[vertex]["z"] = vertex_2_center[2]

        upper_joint = self.sim.getObject(f'/{self.name}/RR_upper_leg_joint')
        lower_joint = self.sim.getObject(f'/{self.name}/RR_lower_leg_joint')
        foot_joint = self.sim.getObject(f'/{self.name}/RR_foot_joint')
        _, self.upper_leg_lenght, _ = self.sim.getObjectPosition(lower_joint,upper_joint)
        _, self.lower_leg_lenght, _ = self.sim.getObjectPosition(foot_joint,lower_joint)

        
    
    def input_speed_actions(self,actions):
        for i, jointName in enumerate(self.joint_list):
            joint = self.handleDict[jointName]
            action_idx = i // 2
            self.sim.setJointTargetVelocity(joint, float(actions[action_idx]))

    def input_position_actions(self,actions):
        for i, jointName in enumerate(self.joint_list):
            joint = self.handleDict[jointName]
            self.sim.setJointTargetPosition(joint, float(actions[i]))

    
    # def get_x_pos(self):
    #     x,y,z = self.get_relative_position()
    #     return x

    # def get_y_pos(self):
    #     x,y,z = self.get_relative_position()
    #     return y

    def get_relative_position(self):
        return self.sim.getObjectPosition(self.robot, self.target)
    def get_orientation(self):
        return self.sim.getObjectOrientation(self.robot, -1)
    def get_velocities(self):
        return self.sim.getObjectVelocity(self.robot)
    def get_joint_information(self,jointName):
        joint = self.handleDict[jointName]
        angle = self.sim.getJointPosition(joint)
        _, speed = self.sim.getObjectFloatParameter(joint, self.sim.jointfloatparam_velocity)
        return [angle,speed]

    def get_joint_speed_0(self,speed):
        return speed < 0.2

    def get_joints_on_max(self):
        joints_data = []
        for jointName in self.joint_list:
            angle,speed = self.get_joint_information(jointName)
            if "upper" in jointName:
                if angle < -1.5 or angle > 1.5:
                    joints_data.append(1)
                else:
                    joints_data.append(0)
            elif "lower" in jointName:
                if angle > -0.1 or angle < -2.3:
                    joints_data.append(1)
                else:
                    joints_data.append(0)
            else:
                raise ValueError("Não foram observadas todas as juntas")
        return joints_data


    def get_all_joints_information(self):
        joints_data = []
        for jointName in self.joint_list:
            joints_data.extend(self.get_joint_information(jointName))
        return joints_data
    
    def get_joints_speed(self):
        speeds = []
        for jointName in self.joint_list:
            joint = self.handleDict[jointName]
            _, speed = self.sim.getObjectFloatParameter(joint, self.sim.jointfloatparam_velocity)
            speeds.append(speed)
        return speeds
    
    def get_joints_position(self):
        positions = []
        for jointName in self.joint_list:
            joint = self.handleDict[jointName]
            angle = self.sim.getJointPosition(joint)
            positions.append(angle)
        return np.array(positions)
    
    def get_joints_angle_change(self):
        new_pos = self.get_joints_position()
        delta = new_pos - self.last_joints_postion
        self.last_joints_postion = new_pos
        return delta

    def check_upside_down(self):
        homo_matrix = self.get_cg()
        if homo_matrix[2][2] < 0:
            return True
        return False

    def check_fall(self):
        return self.z < 0.15

    def check_arrival(self):
        return abs(self.x) < 1.0
    
    def check_vel_0(self):
        [vx, vy, vz], [wx, wy, wz] = self.get_velocities()
        return vx < 0.1
    
    def get_poligon_area(self):
        x1 = self.base_point_polygons["FL"][0]
        y1 = self.base_point_polygons["FL"][1]
        x2 = self.base_point_polygons["FR"][0]
        y2 = self.base_point_polygons["FR"][1]
        x3 = self.base_point_polygons["RR"][0]
        y3 = self.base_point_polygons["RR"][1]
        x4 = self.base_point_polygons["RL"][0]
        y4 = self.base_point_polygons["RL"][1]
        area = 1/2 * abs(x1*y2+x2*y3+x3*y4+x4*y1 - (y1*x2+y2*x3+y3*x4+y4*x1))
        return area



    def check_cg_inside(self,tolerance=1):
        points_dict = self.base_point_polygons
        edges = (
            (points_dict["RL"],points_dict["RR"]),
            (points_dict["RR"],points_dict["FR"]),
            (points_dict["FR"],points_dict["FL"]),
            (points_dict["FL"],points_dict["RL"]),
            )
        cnt = 0
        cg = self.get_cg()
        xp = cg[0][3]
        yp = cg[1][3]
        for edge in edges:
            np_arr = np.array(edge)
            tolerance_edge = np_arr * tolerance
            (x1,y1),(x2,y2) = tolerance_edge
            if (yp<y1) != (yp<y2) and xp < x1 + ((yp-y1)/(y2-y1))*(x2-x1):
                cnt += 1
        return cnt%2 == 1

    def discover_base_polygon_points(self):
        rl_matrix = self.get_joint_final_matrix("RL")
        rr_matrix = self.get_joint_final_matrix("RR")
        fl_matrix = self.get_joint_final_matrix("FL")
        fr_matrix = self.get_joint_final_matrix("FR")

        return {"RL":(rl_matrix[0][3],rl_matrix[1][3]),
                "RR":(rr_matrix[0][3],rr_matrix[1][3]),
                "FL":(fl_matrix[0][3],fl_matrix[1][3]),
                "FR":(fr_matrix[0][3],fr_matrix[1][3])}

    def get_same_side_foot_distance(self):
        rl_matrix = self.get_joint_final_matrix("RL")
        rr_matrix = self.get_joint_final_matrix("RR")
        fl_matrix = self.get_joint_final_matrix("FL")
        fr_matrix = self.get_joint_final_matrix("FR")

        r_distance = math.sqrt((fr_matrix[0][3]-rr_matrix[0][3])**2+(fr_matrix[1][3]-rr_matrix[1][3])**2+(fr_matrix[2][3]-rr_matrix[2][3])**2)
        l_distance = math.sqrt((fl_matrix[0][3]-rl_matrix[0][3])**2+(fl_matrix[1][3]-rl_matrix[1][3])**2+(fl_matrix[2][3]-rl_matrix[2][3])**2)
        return [r_distance,l_distance]
    

    def get_crossing_foot(self):
        rl_matrix = self.get_joint_final_matrix("RL",True)
        rr_matrix = self.get_joint_final_matrix("RR",True)
        fl_matrix = self.get_joint_final_matrix("FL",True)
        fr_matrix = self.get_joint_final_matrix("FR",True)

        return [rr_matrix[0][3] > fr_matrix[0][3],rl_matrix[0][3] > fl_matrix[0][3] ]


    def get_joint_final_matrix(self,vertex,inertial_reference=False):
        dx = self.vertex_2_center[vertex]["x"]
        dy = self.vertex_2_center[vertex]["y"]
        dz = self.vertex_2_center[vertex]["z"]
        theta_1 = self.sim.getJointPosition(self.sim.getObject(f'/{self.name}/{vertex}_upper_leg_joint'))
        theta_2 = self.sim.getJointPosition(self.sim.getObject(f'/{self.name}/{vertex}_lower_leg_joint'))

        matrix = self.discover_foot_position(
            dx,dy,dz,\
            self.upper_leg_lenght,self.lower_leg_lenght,\
            theta_1,theta_2,inertial_reference
            )
        return matrix

    def get_elbow_final_matrix(self,vertex,inertial_reference=False):
            dx = self.vertex_2_center[vertex]["x"]
            dy = self.vertex_2_center[vertex]["y"]
            dz = self.vertex_2_center[vertex]["z"]
            theta_1 = self.sim.getJointPosition(self.sim.getObject(f'/{self.name}/{vertex}_upper_leg_joint'))
            theta_2 = self.sim.getJointPosition(self.sim.getObject(f'/{self.name}/{vertex}_lower_leg_joint'))

            matrix = self.discover_elbow_position(
                dx,dy,dz,\
                self.upper_leg_lenght,\
                theta_1,theta_2,inertial_reference
                )
            return matrix

    def discover_foot_position(self,dx,dy,dz,upper_length,lower_length,theta_1,theta_2,inertial_reference=False):
        r_3_4 = self.get_homgeneous_transform_matrix(0,0,lower_length,0,"x")

        if inertial_reference:
            return (self.discover_elbow_position(dx,dy,dz,upper_length,theta_1,theta_2,inertial_reference) @ r_3_4)
        return self.inverse_rotation_matrix @ (self.discover_elbow_position(dx,dy,dz,upper_length,theta_1,theta_2,inertial_reference) @ r_3_4)
    
    def discover_elbow_position(self,dx,dy,dz,upper_length,theta_1,theta_2,local_reference=False):
        r_0_1 = self.get_homgeneous_transform_matrix(theta_1,dx,dy,dz,"y")
        r_1_2 = self.get_homgeneous_transform_matrix(-math.pi/2,0,0,0,"x")
        r_2_3 = self.get_homgeneous_transform_matrix(theta_2,0,upper_length,0,"z")

        if local_reference:
            return (((r_0_1 @ r_1_2) @ r_2_3))

        return self.inverse_rotation_matrix @ (((r_0_1 @ r_1_2) @ r_2_3))
    

    def get_homgeneous_transform_matrix(self,theta,dx,dy,dz,axis):
        c = np.cos(theta)
        s = np.sin(theta)
        if axis == "x":
            matrix = np.array([
                [1, 0, 0,dx],
                [0, c,-s,dy],
                [0, s, c,dz],
                [0, 0, 0, 1]])
        if axis == "y":
            matrix = np.array([
                [ c, 0, s,dx],
                [ 0, 1, 0,dy],
                [-s, 0, c,dz],
                [ 0, 0, 0, 1]])
        if axis == "z":
            matrix = np.array([
                [c,-s, 0,dx],
                [s, c, 0,dy],
                [0, 0, 1,dz],
                [0, 0, 0, 1]])
        return matrix
    
    def get_cg(self):
        total_mass = 0
        total_cog = [0, 0, 0]
        for link in self.links_list:
            link = self.sim.getObject(f'/Doggy/{link}/')
            mass, inertia, cog = self.sim.getShapeMassAndInertia(link, self.sim.getObjectMatrix(self.robot,-1))
            total_mass += mass
            total_cog = [total_cog[i] + mass * cog[i] for i in range(3)]
        total_cog = [c / total_mass for c in total_cog]

        homo_matrix = np.array([[1,0,0,total_cog[0]],
                                [0,1,0,total_cog[1]],
                                [0,0,1,total_cog[2]],
                                [0,0,0,1]])

        return homo_matrix

    def get_inverse_rotation_matrix(self):
        inv_roll = self.get_homgeneous_transform_matrix(self.roll,0,0,0,"x")
        inv_pitch = self.get_homgeneous_transform_matrix(self.pitch,0,0,0,"y")
        inv_yaw = self.get_homgeneous_transform_matrix(self.yaw,0,0,0,"z")
        return inv_roll @ inv_pitch @ inv_yaw

    def get_correct_direction_angle(self):
        loss_angle = math.atan2(-self.y, -self.x)
        delta = loss_angle - self.yaw
        delta = (delta + math.pi) % (2 * math.pi) - math.pi
        return delta
    

    def get_feet_above_base_link(self):
        positions = ["RL","RR","FL","FR"]
        height_ref = [(self.inverse_rotation_matrix @ self.foots_matrix[position])[2][3] for position in positions]
        count = sum(h > 0 for h in height_ref)
        return count

    def get_num_elbows_above_feet(self):
        positions = ["RL","RR","FL","FR"]
        counter = 0
        for position in positions:
            feet_h = self.get_joint_final_matrix(position,False)[2][3]
            elbow_h = self.get_elbow_final_matrix(position,False)[2][3]
            if feet_h < elbow_h:
                counter += 1
        return counter

        

        
