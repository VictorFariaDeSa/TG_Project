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
        self.robot = self.sim.getObject(f'/{self.name}/')
        self.target = self.sim.getObject(f'/{self.target_name}')
        self.fill_robot_data()
        self.last_x = self.sim.getObjectPosition(self.robot, self.target)[0]
        self.last_joints_postion = self.get_joints_position()
        self.last_joints_orientation = self.get_joints_orientation()
        self.last_time = self.sim.getSimulationTime()
        self.last_joints_speed = self.get_joints_speed()




    def get_joints_acceleration(self):
        joints_speed = self.get_joints_speed()
        delta_speeds = np.array(joints_speed)-np.array(self.last_joints_speed)
        curr_time = self.sim.getSimulationTime()
        delta_time = curr_time - self.last_time
        accel = delta_speeds/delta_time

        self.last_time = curr_time
        self.last_joints_speed = joints_speed




        return accel

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



    def get_x_pos(self):
        x,y,z = self.get_relative_position()
        return x

    def get_y_pos(self):
        x,y,z = self.get_relative_position()
        return y

    def get_relative_position(self):
        return self.sim.getObjectPosition(self.robot, self.target)
    def get_orientation(self):
        return self.sim.getObjectOrientation(self.robot, -1)
    def get_velocities(self):
        return self.sim.getObjectVelocity(self.robot)
    def get_matrix(self):
        return self.sim.getObjectMatrix(self.robot, -1)
    def get_joint_information(self,jointName):
        joint = self.handleDict[jointName]
        angle = self.sim.getJointPosition(joint)
        _, speed = self.sim.getObjectFloatParameter(joint, self.sim.jointfloatparam_velocity)
        return [angle,speed]
    

    def get_joints_orientation_change(self):
        new_orientation = self.get_joints_orientation()
        n_changes = np.sum(new_orientation != self.last_joints_orientation)
        self.last_joints_orientation = new_orientation
        return n_changes

    def get_joints_orientation(self):
        speeds = np.array(self.get_joints_speed())
        orientation = np.where(speeds > 0, 1, np.where(speeds < 0, -1, 0))
        return orientation

    def get_joints_speed_0(self):
        speeds = self.get_joints_speed()
        check_zero_speed = [0 if speed > 0.2 else 1 for speed in speeds]
        return check_zero_speed

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

    def check_stability(self,max_deg):
        roll, pitch, yaw = self.get_orientation()
        stable = abs(roll) < max_deg and abs(pitch) < max_deg and abs(yaw) < max_deg
        return stable

    def get_delta_x(self):
        new_x,new_y,new_z = self.get_relative_position()
        delta_x = abs(self.last_x) - abs(new_x)
        self.last_x = new_x
        return delta_x
    
    def check_fall(self):
        _, _, z = self.get_relative_position()
        return z < 0.15

    def check_arrival(self):
        dx, _, _ = self.get_relative_position()
        return abs(dx) < 1.0
    
    def check_vel_0(self):
        [vx, vy, vz], [wx, wy, wz] = self.get_velocities()
        return vx < 0.1
    
    def get_poligon_area(self):
        points_dict = self.discover_base_polygon_points()
        x1 = points_dict["FL"][0]
        y1 = points_dict["FL"][1]
        x2 = points_dict["FR"][0]
        y2 = points_dict["FR"][1]
        x3 = points_dict["RR"][0]
        y3 = points_dict["RR"][1]
        x4 = points_dict["RL"][0]
        y4 = points_dict["RL"][1]
        area = 1/2 * abs(x1*y2+x2*y3+x3*y4+x4*y1 - (y1*x2+y2*x3+y3*x4+y4*x1))
        return area



    def cg_inside(self,tolerance=1):
        points_dict = self.discover_base_polygon_points()
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


    def discover_foot_position(self,dx,dy,dz,upper_length,lower_length,theta_1,theta_2,inertial_reference=False):
        roll, pitch, yaw = self.sim.getObjectOrientation(self.robot, -1)
        inv_roll = self.get_homgeneous_transform_matrix(roll,0,0,0,"x")
        inv_pitch = self.get_homgeneous_transform_matrix(pitch,0,0,0,"y")
        inv_yaw = self.get_homgeneous_transform_matrix(yaw,0,0,0,"z")
        inverse_rotation_matrix = inv_roll @ inv_pitch @ inv_yaw
        
        r_0_1 = self.get_homgeneous_transform_matrix(theta_1,dx,dy,dz,"y")
        r_1_2 = self.get_homgeneous_transform_matrix(-math.pi/2,0,0,0,"x")
        r_2_3 = self.get_homgeneous_transform_matrix(theta_2,0,upper_length,0,"z")
        r_3_4 = self.get_homgeneous_transform_matrix(0,0,lower_length,0,"x")

        if inertial_reference:
            return (((r_0_1 @ r_1_2) @ r_2_3) @ r_3_4)

        return inverse_rotation_matrix @ (((r_0_1 @ r_1_2) @ r_2_3) @ r_3_4)
    

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
        
        
        roll, pitch, yaw = self.sim.getObjectOrientation(self.robot, -1)
        inv_roll = self.get_homgeneous_transform_matrix(roll,0,0,0,"x")
        inv_pitch = self.get_homgeneous_transform_matrix(pitch,0,0,0,"y")
        inv_yaw = self.get_homgeneous_transform_matrix(yaw,0,0,0,"z")
        inverse_rotation_matrix = inv_roll @ inv_pitch @ inv_yaw

        homo_matrix = np.array([[1,0,0,total_cog[0]],
                                [0,1,0,total_cog[1]],
                                [0,0,1,total_cog[2]],
                                [0,0,0,1]])

        return inverse_rotation_matrix @ homo_matrix




    def get_correct_direction_angle(self):
        x,y,z = self.get_relative_position()
        roll, pitch,yaw = self.get_orientation()
        loss_angle = math.atan2(-y, -x)
        delta = loss_angle - yaw
        delta = (delta + math.pi) % (2 * math.pi) - math.pi
        return delta
    

    def get_feet_above_base_link(self):
        positions = ["RL","RR","FL","FR"]
        height_ref = [self.get_joint_final_matrix(position,True)[2][3] for position in positions]
        count = sum(h > 0 for h in height_ref)
        return count

        

        
