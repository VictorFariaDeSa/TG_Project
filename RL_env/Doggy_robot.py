import numpy as np
import math
import copy
class Doggy_robot():
    def __init__(self,sim,robot_name,target_name):
        self.sim = sim
        self.name = robot_name
        self.target_name = target_name
        self.sensor = sim.getObject(f'./{robot_name}/base_link_visual/LaserPointer/sensor')
        self.vertex = ["RL","RR","FL","FR"]
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


        self.foot_handles = {}
        self.floor_handles = [self.sim.getObjectHandle(f"./Floor[{i}]") for i in range(12)]
        for vertex in self.vertex:
            foot_path = f"./{self.name}/{vertex}_upper_leg_joint/{vertex}_lower_leg_joint/{vertex}_foot_joint/{vertex}_foot_link_respondable"
            self.foot_handles[vertex] = self.sim.getObjectHandle(foot_path)
        self.joints_target = np.zeros(8)
        self.last_joints_target = np.zeros(8)
        self.last_last_joints_target = np.zeros(8)



        self.fill_robot_data()
        
        self.joints_data = {}
        self.fill_joints_dict()

        self.last_joints_postion = self.get_joints_position()
        self.last_x = self.get_relative_position()[0]
        self.last_joints_orientation = self.get_joints_orientation()
        self.last_time = self.sim.getSimulationTime()
        self.last_joints_speed = self.get_joints_speeds()
        self.last_positions = self.get_relative_position()

        self.elbows_HT_matrix = {}
        self.elbows_inertial_HT_matrix = {}
        self.foots_HT_matrix = {}
        self.foots_inertial_HT_matrix = {}

        self.last_foot_HT_matrix ={}
        

        self.update_robot_data()

    def read_laser_height(self):
        og_vector = np.array([[0],[0],[self.sim.readProximitySensor(self.sensor)[1]+0.1430],[1]])
        new_height_vector = self.inverse_matrix @ og_vector
        return new_height_vector

    def fill_robot_data(self):
        handleDict = {}
        for name in self.joint_list:
            handleDict[name] = self.sim.getObject(f"/{self.name}/{name}")
        self.handleDict = handleDict


        self.vertex_2_center = {}
        for vertex in self.vertex:
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

    def input_speed_actions(self,actions):
        for i, jointName in enumerate(self.joint_list):
            joint = self.handleDict[jointName]
            self.sim.setJointTargetVelocity(joint, float(actions[i]))

    def input_position_actions(self,actions):
        for i, jointName in enumerate(self.joint_list):
            joint = self.handleDict[jointName]
            self.sim.setJointTargetPosition(joint, float(actions[i]))
        self.last_last_joints_target = self.last_joints_target.copy() if self.last_joints_target is not None else None
        self.last_joints_target = self.joints_target.copy() if self.joints_target is not None else None
        self.joints_target = np.array(actions)

    def input_torque_actions(self, actions):
        print(actions)
        for i, jointName in enumerate(self.joint_list):
            joint = self.handleDict[jointName]
            self.sim.setJointMode(joint, self.sim.jointmode_force, 0)
            self.sim.setJointForce(joint, float(actions[i]*10))

    def update_robot_data(self):
        self.positions = self.get_relative_position()
        self.delta_x = abs(self.last_x) - abs(self.positions[0])
        self.orientations = self.get_orientation()
        [self.linear_velocities,self.angular_velocities] = self.get_velocities()
        self.inverse_matrix = self.get_inverse_rotation_matrix()
        self.cg_matrix = self.get_cg()
        self.inertial_cg_matrix = self.inverse_matrix @ self.cg_matrix
        self.time = self.sim.getSimulationTime()
        for vertex in self.vertex:
            self.elbows_HT_matrix[vertex] = self.get_elbow_final_matrix(vertex)
            self.elbows_inertial_HT_matrix[vertex] = self.inverse_matrix @ self.elbows_HT_matrix[vertex]
            self.foots_HT_matrix[vertex] = self.discover_foot_position(self.elbows_HT_matrix[vertex],self.lower_leg_lenght)
            self.foots_inertial_HT_matrix[vertex] = self.inverse_matrix @ self.foots_HT_matrix[vertex]
        self.fill_joints_dict()
        self.height = self.read_laser_height()[2]
        self.foot_colision = self.check_foot_collision()
        # print(f"z: {self.positions[2]} ---- height: {self.height} ---- diff: {self.height - self.positions[2]}")

        self.fall = self.check_fall()
        self.upside_down = self.check_upside_down()
        self.arrival = self.check_arrival()
        self.last_x = self.positions[0]




    def check_foot_collision(self):
        foot_collision = {}
        for vertex, foot_handle in self.foot_handles.items():
            contact = 0
            for floor_handle in self.floor_handles:
                if self.sim.checkCollision(foot_handle, floor_handle)[0]:
                    contact = 1
                    break  # já encontrou contato, não precisa testar o resto
            foot_collision[vertex] = contact
        return foot_collision



    def get_relative_position(self):
        return self.sim.getObjectPosition(self.robot, self.target)

    def get_orientation(self):
        return self.sim.getObjectOrientation(self.robot, -1)
   
    def get_velocities(self):
        return self.sim.getObjectVelocity(self.robot)

    def get_inverse_rotation_matrix(self):
        roll, pitch, yaw = self.orientations
        inv_roll = self.get_homgeneous_transform_matrix(roll,0,0,0,"x")
        inv_pitch = self.get_homgeneous_transform_matrix(pitch,0,0,0,"y")
        inv_yaw = self.get_homgeneous_transform_matrix(yaw,0,0,0,"z")
        inverse_rotation_matrix = inv_roll @ inv_pitch @ inv_yaw
        return inverse_rotation_matrix

    def check_upside_down(self):
        return self.inverse_matrix[2][2] < 0

    def check_fall(self):
        _, _, z = self.positions
        return z < 0.15

    def check_arrival(self):
        dx, dy, _ = self.positions
        return abs(dx) < 1.0 and abs(dy) < 1.0
    
    def check_vel_0(self):
        return self.linear_velocities[0] < 0.1

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

    def discover_elbow_position(self,dx,dy,dz,upper_length,theta_1,theta_2): 
        r_0_1 = self.get_homgeneous_transform_matrix(theta_1,dx,dy,dz,"y")
        r_1_2 = self.get_homgeneous_transform_matrix(-math.pi/2,0,0,0,"x")
        r_2_3 = self.get_homgeneous_transform_matrix(theta_2,0,upper_length,0,"z")

        return (((r_0_1 @ r_1_2) @ r_2_3))

    def get_elbow_final_matrix(self,vertex):
            dx = self.vertex_2_center[vertex]["x"]
            dy = self.vertex_2_center[vertex]["y"]
            dz = self.vertex_2_center[vertex]["z"]
            theta_1 = self.sim.getJointPosition(self.sim.getObject(f'/{self.name}/{vertex}_upper_leg_joint'))
            theta_2 = self.sim.getJointPosition(self.sim.getObject(f'/{self.name}/{vertex}_lower_leg_joint'))

            matrix = self.discover_elbow_position(
                dx,dy,dz,\
                self.upper_leg_lenght,\
                theta_1,theta_2
                )
            return matrix

    def discover_foot_position(self,elbow_HT_matrix,lower_length):
        r_3_4 = self.get_homgeneous_transform_matrix(0,0,lower_length,0,"x")
        return elbow_HT_matrix @ r_3_4
    
    def get_num_elbows_above_feet(self):
        counter = 0
        for position in self.vertex:
            feet_h = self.foots_HT_matrix[position][2][3]
            elbow_h = self.elbows_HT_matrix[position][2][3]
            if feet_h < elbow_h:
                counter += 1
        return counter

    def get_crossing_foot(self):
        rl_matrix = self.foots_inertial_HT_matrix["RL"]
        rr_matrix = self.foots_inertial_HT_matrix["RR"]
        fl_matrix = self.foots_inertial_HT_matrix["FL"]
        fr_matrix = self.foots_inertial_HT_matrix["FR"]

        return [rr_matrix[0][3] > fr_matrix[0][3],rl_matrix[0][3] > fl_matrix[0][3] ]

    def get_same_side_foot_distance(self):
        rl_matrix = self.foots_HT_matrix["RL"]
        rr_matrix = self.foots_HT_matrix["RR"]
        fl_matrix = self.foots_HT_matrix["FL"]
        fr_matrix = self.foots_HT_matrix["FR"]

        r_distance = math.sqrt((fr_matrix[0][3]-rr_matrix[0][3])**2+(fr_matrix[1][3]-rr_matrix[1][3])**2+(fr_matrix[2][3]-rr_matrix[2][3])**2)
        l_distance = math.sqrt((fl_matrix[0][3]-rl_matrix[0][3])**2+(fl_matrix[1][3]-rl_matrix[1][3])**2+(fl_matrix[2][3]-rl_matrix[2][3])**2)
        return [r_distance,l_distance]

    def get_joint_information(self,jointName):
        joint = self.handleDict[jointName]
        angle = self.sim.getJointPosition(joint)
        _, speed = self.sim.getObjectFloatParameter(joint, self.sim.jointfloatparam_velocity)
        return [angle,speed]

    def check_joint_maxed(self,jointName,jointAngle):
        if "upper" in jointName:
            if jointAngle < -1.5 or jointAngle > 1.5:
                return 1
            else:
                return 0
        elif "lower" in jointName:
            if jointAngle > -0.1 or jointAngle < -2.3:
                return 1
            else:
                return 0

    def get_joint_force(self,jointName):
        joint = self.handleDict[jointName]
        force = self.sim.getJointForce(joint)
        return force

    def fill_joints_dict(self):
        for jointName in self.joint_list:
            angle,speed = self.get_joint_information(jointName)
            maxed = self.check_joint_maxed(jointName,angle)
            force = self.get_joint_force(jointName)
            self.joints_data[jointName]={}
            self.joints_data[jointName]["angle"] = angle
            self.joints_data[jointName]["speed"] = speed
            self.joints_data[jointName]["maxed"] = maxed
            self.joints_data[jointName]["force"] = force


    def get_joints_speeds(self):
        speeds = []
        for jointName in self.joint_list:
            speeds.append(self.joints_data[jointName]["speed"])
        return speeds
    
    def get_joints_position(self):
        positions = []
        for jointName in self.joint_list:
            positions.append(self.joints_data[jointName]["angle"])
        return np.array(positions)

    def get_joints_on_max(self):
        maxed = []
        for jointName in self.joint_list:
            maxed.append(self.joints_data[jointName]["maxed"])
        return maxed

    def get_joints_torque(self):
        force = []
        for jointName in self.joint_list:
            force.append(self.joints_data[jointName]["force"])
        return force


    def get_all_joints_information(self):
        joints_data = []
        for jointName in self.joint_list:
            joints_data.extend([self.joints_data[jointName]["angle"],self.joints_data[jointName]["speed"]])
        return joints_data
    
    def get_feet_above_base_link(self):
        height_ref = [self.foots_inertial_HT_matrix[vertex][2][3] for vertex in self.vertex]
        count = sum(h > 0 for h in height_ref)
        return count

    def discover_base_polygon_points(self):
        rl_matrix = self.foots_inertial_HT_matrix["RL"]
        rr_matrix = self.foots_inertial_HT_matrix["RR"]
        fl_matrix = self.foots_inertial_HT_matrix["FL"]
        fr_matrix = self.foots_inertial_HT_matrix["FR"]

        return {"RL":(rl_matrix[0][3],rl_matrix[1][3]),
                "RR":(rr_matrix[0][3],rr_matrix[1][3]),
                "FL":(fl_matrix[0][3],fl_matrix[1][3]),
                "FR":(fr_matrix[0][3],fr_matrix[1][3])}

    def cg_inside(self,tolerance=1):
        points_dict = self.discover_base_polygon_points()
        edges = (
            (points_dict["RL"],points_dict["RR"]),
            (points_dict["RR"],points_dict["FR"]),
            (points_dict["FR"],points_dict["FL"]),
            (points_dict["FL"],points_dict["RL"]),
            )
        cnt = 0
        cg = self.inertial_cg_matrix
        xp = cg[0][3]
        yp = cg[1][3]
        for edge in edges:
            np_arr = np.array(edge)
            tolerance_edge = np_arr * tolerance
            (x1,y1),(x2,y2) = tolerance_edge
            if (yp<y1) != (yp<y2) and xp < x1 + ((yp-y1)/(y2-y1))*(x2-x1):
                cnt += 1
        return cnt%2 == 1

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

    def get_correct_direction_angle(self):
        x,y,z = self.positions
        roll, pitch,yaw = self.orientations
        loss_angle = math.atan2(-y, -x)
        delta = loss_angle - yaw
        delta = (delta + math.pi) % (2 * math.pi) - math.pi
        return delta


    def get_delta_foot_positions(self):
        total_displacement = 0
        if (len(self.last_foot_HT_matrix.keys())>0):
            delta_dict = {}
            for key in self.last_foot_HT_matrix.keys():
                delta_dict[key] = self.foots_HT_matrix[key] - self.last_foot_HT_matrix[key]
                dx = self.foots_HT_matrix[key][0][3] - self.last_foot_HT_matrix[key][0][3]
                dy = self.foots_HT_matrix[key][1][3] - self.last_foot_HT_matrix[key][1][3]
                dz = self.foots_HT_matrix[key][2][3] - self.last_foot_HT_matrix[key][2][3]
                total_displacement += abs(dx) + abs(dy) + abs(dz)
        self.last_foot_HT_matrix = copy.deepcopy(self.foots_HT_matrix)
        return total_displacement

    def get_joints_delta_x_pos(self):
        frequency = 1 
        amplitude = 0.15
        delta_positions = []
        for vertex in self.vertex:
            offset = math.pi/2 if vertex in ["RL","FR"] else 0
            pos = self.vertex_2_center[vertex]["x"] + (amplitude * math.sin(math.pi*frequency+offset))
            delta = self.foots_inertial_HT_matrix[vertex][0][3] - pos
            delta_positions.append(delta)
        return delta_positions







    def get_joints_orientation(self):
        speeds = np.array(self.get_joints_speeds())
        orientation = np.where(speeds > 0, 1, np.where(speeds < 0, -1, 0))
        return orientation

    def get_joints_orientation_change(self):
        new_orientation = self.get_joints_orientation()
        n_changes = np.sum(new_orientation != self.last_joints_orientation)
        self.last_joints_orientation = new_orientation
        return n_changes





    def get_joints_acceleration(self):
        joints_speed = self.get_joints_speeds()
        delta_speeds = np.array(joints_speed)-np.array(self.last_joints_speed)
        delta_time = self.time - self.last_time
        accel = delta_speeds/delta_time

        self.last_time = self.time
        self.last_joints_speed = joints_speed

        return accel





    def get_joints_speed_0(self):
        speeds = self.get_joints_speeds()
        check_zero_speed = [0 if speed > 0.2 else 1 for speed in speeds]
        return check_zero_speed


    def get_joints_angle_change(self):
        new_pos = self.get_joints_position()
        delta = new_pos - self.last_joints_postion
        self.last_joints_postion = new_pos
        return delta



    def get_desired_xy_pos(self,vertex,t,freq,k1,k2):
        
        if vertex in ["RL", "FR"]:
            offset = math.pi
        elif vertex in ["FL", "RR"]:
            offset = 0
        else:
            raise ValueError(f"Erro: jointName inválido: {vertex}")


        x = k1 * np.cos(freq/math.pi*t + offset)
        y = k2 * np.sin(freq/math.pi*t + offset)
        if y > 0:
            y=0
        return (x+0.1,y+0.35)
    
    def inverse_kin(self,coords,upper_len,lower_len):
        x = coords[0]
        y = coords[1]

        if (x**2 + y**2)>(upper_len+lower_len)**2:
            print("Fora do espaço")

        D = (x**2+y**2-lower_len**2-upper_len**2)/(2*upper_len*lower_len)
        theta2 = math.atan2(math.sqrt(1-D**2),D)#revisar o sinal do -
        gamma = math.atan2(lower_len*math.sin(theta2),upper_len + lower_len*math.cos(theta2))
        theta1 = math.atan2(y,x) - gamma

        return (theta1,theta2)
    
    def get_thetas_dict(self):
        thetas_dict = {}
        for vertex in self.vertex:
            thetas_dict[vertex] = {}
            pos = self.get_desired_xy_pos(vertex,self.time,40,0.25,0.15)
            thetas = self.inverse_kin(pos,self.upper_leg_lenght,self.lower_leg_lenght)
            thetas_dict[vertex]["theta1"] = thetas[0]
            thetas_dict[vertex]["theta2"] = thetas[1]
        return thetas_dict
    
    def convert_thetas_dict_to_actions_list(self,theta_dict):
        actions = []
        for jointName in self.joint_list:
            thetas = theta_dict[jointName[0:2]]
            if "upper" in jointName:
                actions.append(thetas["theta1"])
            else:
                actions.append(-thetas["theta2"])
        return actions










    





        

        

    

