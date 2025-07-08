"""
mqtt_controller.py - MQTT communication for telescope control
"""
import os
import json
import time
import logging
import paho.mqtt.client as mqtt
from datetime import datetime, timezone
from tracking.utils.config import config
from tracking.utils.colors import Colors

# Get logger for this module
logger = logging.getLogger(__name__)

class MqttController:
    def __init__(self):
        """
        Initialize the MqttController object with default parameters.
        
        Sets up MQTT topics from configuration.
        """
        # MQTT setup
        self.mqtt_client = None
        self.mqtt_connected = False
        self.ant = None
        
        # Topics from config
        self.topic_az_status = config.mqtt.topic_az_status
        self.topic_az_cmd = config.mqtt.topic_az_cmd
        self.topic_el_status = config.mqtt.topic_el_status
        self.topic_el_cmd = config.mqtt.topic_el_cmd
        self.topic_prg_trk_cmd = config.mqtt.topic_prg_trk_cmd
        self.topic_act_time = config.mqtt.topic_act_time
        self.topic_start_time = config.mqtt.topic_start_time

        # Messages
        self.messages = {}

    def on_connect(self, client, userdata, flags, rc):
        """
        MQTT connection callback function.
        
        Called when the client receives a CONNACK response from the server.
        
        Args:
            client: The client instance for this callback
            userdata: The private user data as set in Client() or userdata_set()
            flags: Response flags sent by the broker
            rc (int): The connection result code
                0: Connection successful
                1: Connection refused - incorrect protocol version
                2: Connection refused - invalid client identifier
                3: Connection refused - server unavailable
                4: Connection refused - bad username or password
                5: Connection refused - not authorized
        """
        if rc == 0:
            logger.info(f"{Colors.GREEN}Connected to MQTT broker{Colors.RESET}")
            self.mqtt_connected = True
        else:
            logger.error(f"{Colors.RED}Failed to connect, return code {rc}{Colors.RESET}")
            self.mqtt_connected = False
            
    def on_message(self, client, userdata, msg):
        """
        MQTT message callback function.
        
        Called when a message has been received on a topic that the client subscribes to.
        Stores the message payload in the messages dictionary indexed by topic.
        
        Args:
            client: The client instance for this callback
            userdata: The private user data as set in Client() or userdata_set()
            msg: An instance of MQTTMessage containing topic and payload
        """
        try:
            # Parse JSON message and store it
            data = json.loads(msg.payload.decode())
            self.messages[msg.topic] = data
            logger.debug(f"Received message on {msg.topic}: {data}")
        except json.JSONDecodeError as e:
            logger.warning(f"{Colors.RED}Failed to parse MQTT message: {e}{Colors.RESET}")
        except Exception as e:
            logger.error(f"{Colors.RED}Error processing MQTT message: {e}{Colors.RESET}")

    def setup_mqtt(self, ant, port=None):
        """
        Set up MQTT client and establish connection to the broker.
        
        Configures the MQTT client with callbacks, connects to the appropriate broker
        based on antenna selection, and subscribes to all necessary topics.
        
        Args:
            ant (str): Antenna identifier - "N" for North or "S" for South
            port (int): MQTT broker port number (default: from config)
            
        Raises:
            ValueError: If antenna identifier is not "N" or "S"
        """
        if port is None:
            port = config.mqtt.port

        if ant not in ["N", "S"]:
            raise ValueError(f"Invalid antenna '{ant}'. Must be 'N' or 'S'.")

        self.ant = ant

        if ant == "N":
            broker_ip = config.mqtt.north_broker_ip
        elif ant == "S":
            broker_ip = config.mqtt.south_broker_ip
        else:
            raise ValueError("How did tis happen?")

        self.mqtt_client = mqtt.Client(client_id=config.mqtt.client_id)
        self.mqtt_client.on_connect = self.on_connect
        self.mqtt_client.on_message = self.on_message
        
        # Connect to broker
        self.mqtt_client.connect(broker_ip, port, config.mqtt.connection_timeout)
        self.mqtt_client.loop_start()
        
        # Wait for connection
        timeout = config.mqtt.connection_wait_timeout
        start_time = time.time()
        while not self.mqtt_connected and (time.time() - start_time) < timeout:
            time.sleep(config.mqtt.poll_sleep_interval)
        
        if not self.mqtt_connected:
            raise ConnectionError(f"Failed to connect to MQTT broker {broker_ip}:{port} within {timeout}s")
        
        logger.info(f"{Colors.GREEN}Connected: {self.mqtt_connected}{Colors.RESET}")
        
        # Subscribe to topics
        self.mqtt_client.subscribe(self.topic_az_status)
        self.mqtt_client.subscribe(self.topic_az_cmd)
        self.mqtt_client.subscribe(self.topic_el_status)
        self.mqtt_client.subscribe(self.topic_el_cmd)
        self.mqtt_client.subscribe(self.topic_prg_trk_cmd)
        self.mqtt_client.subscribe(self.topic_act_time)
        self.mqtt_client.subscribe(self.topic_start_time)
        
    def send_mqtt_command(self, topic, command_dict):
        """
        Send a JSON-encoded command via MQTT.
        
        Converts a Python dictionary to JSON format and publishes it to the specified topic.
        
        Args:
            topic (str): MQTT topic to publish to
            command_dict (dict): Command data to be JSON-encoded and sent
        """
        if not self.mqtt_connected or self.mqtt_client is None:
            raise ConnectionError("MQTT not connected")
        
        try:
            message = json.dumps(command_dict)
            result = self.mqtt_client.publish(topic, message)
            if result.rc != mqtt.MQTT_ERR_SUCCESS:
                logger.error(f"{Colors.RED}Failed to publish MQTT message: {result.rc}{Colors.RESET}")
        except Exception as e:
            logger.error(f"{Colors.RED}Error sending MQTT command: {e}{Colors.RESET}")
            raise
        
    def read_mqtt_topic(self, topic, timeout=None):
        """
        Read the latest message from a specific MQTT topic.
        
        Clears any previous message for the topic and waits for a new one.
        Returns the JSON-decoded message content.
        
        Args:
            topic (str): MQTT topic to read from
            timeout (float): Maximum time to wait for message in seconds (default: from config)
            
        Returns:
            dict or None: JSON-decoded message content if received within timeout,
                         None if no message received
        """
        if timeout is None:
            timeout = config.mqtt.read_timeout
            
        # Clear previous message
        if topic in self.messages:
            del self.messages[topic]
            
        # Wait for new message
        start_time = time.time()
        while topic not in self.messages and (time.time() - start_time) < timeout:
            time.sleep(config.mqtt.poll_sleep_interval)
            
        if topic in self.messages:
            return self.messages[topic]
        return None
        
    def set_axis_mode_track(self):
        """
        Set both azimuth and elevation axes to tracking mode.
        Sends track commands to both axes with zero velocity and acceleration arguments.
        Waits for the axes to enter 'track' mode, up to a configurable timeout.
        """
        # AZ
        sv = {
            'id_client': config.mqtt.id_client_motion,
            'source': config.mqtt.source,
            'destination': config.mqtt.destination_az,
            'name': 'track',
            'args': [0, 0]
        }
        self.send_mqtt_command(self.topic_az_cmd, sv)
        # EL
        sv = {
            'id_client': config.mqtt.id_client_motion,
            'source': config.mqtt.source,
            'destination': config.mqtt.destination_el,
            'name': 'track',
            'args': [0, 0]
        }
        self.send_mqtt_command(self.topic_el_cmd, sv)
        # Wait for mode
        self._wait_for_mode('track')
        self.print_status()

    def set_axis_mode_stop(self):
        """
        Stop both azimuth and elevation axes.
        Sends stop commands to both axes to halt any ongoing motion.
        Waits for the axes to enter 'stop' mode, up to a configurable timeout.
        """
        # AZ
        sv = {
            'id_client': config.mqtt.id_client_motion,
            'source': config.mqtt.source,
            'destination': config.mqtt.destination_az,
            'name': 'stop',
            'args': []
        }
        self.send_mqtt_command(self.topic_az_cmd, sv)
        # EL
        sv = {
            'id_client': config.mqtt.id_client_motion,
            'source': config.mqtt.source,
            'destination': config.mqtt.destination_el,
            'name': 'stop',
            'args': []
        }
        self.send_mqtt_command(self.topic_el_cmd, sv)
        # Wait for mode
        self._wait_for_mode('stop')
        self.print_status()

    def set_axis_mode_position(self, az, el):
        """
        Set both azimuth and elevation axes to position mode.
        Sends position commands to both axes with zero velocity and acceleration arguments. False indicates that the position is NOT relative.
        Waits for the axes to enter 'position' mode, up to a configurable timeout.
        """
        logger.info(f"{Colors.BLUE}Setting position mode: AZ={az:.2f}°, EL={el:.2f}°{Colors.RESET}")
        # AZ
        sv = {
            'id_client': config.mqtt.id_client_motion,
            'source': config.mqtt.source,
            'destination': config.mqtt.destination_az,
            'name': 'position',
            'args': [False, az, 0, 0]
        }
        self.send_mqtt_command(self.topic_az_cmd, sv)
        # EL
        sv = {
            'id_client': config.mqtt.id_client_motion,
            'source': config.mqtt.source,
            'destination': config.mqtt.destination_el,
            'name': 'position',
            'args': [False, el, 0, 0]
        }
        self.send_mqtt_command(self.topic_el_cmd, sv)
        # Wait for mode
        self._wait_for_mode('position')
        self.print_status()

    def _wait_for_mode(self, target_mode):
        """
        Wait for both axes to reach the target mode, with timeout and poll interval from config.
        """
        timeout = config.telescope.axis_mode_settle_timeout
        poll_interval = config.telescope.axis_mode_poll_interval
        start_time = time.time()
                
        while time.time() - start_time < timeout:
            az_mode, el_mode = self.get_current_mode()
            if az_mode == target_mode or el_mode == target_mode:
                elapsed = time.time() - start_time
                logger.info(f"{Colors.GREEN}Axes put in '{target_mode}' mode in {elapsed:.1f}s{Colors.RESET}")
                return True
            time.sleep(poll_interval)
        
        elapsed = time.time() - start_time
        logger.warning(f"{Colors.RED}Warning: Timeout waiting for axes to be put in '{target_mode}' mode after {elapsed:.1f}s{Colors.RESET}")
        return False

    def print_status(self):
        # AZ status - show only key info
        az_status = self.read_mqtt_topic(self.topic_az_status)
        if az_status and 'v' in az_status:
            v = az_status['v']
            logger.info(f"AZ: state={v.get('state', 'N/A')}, pos={v.get('act_pos', 'N/A'):.2f}°, target={v.get('target_pos', 'N/A'):.2f}°, in_target={v.get('in_target', 'N/A')}")
            
        # EL status - show only key info
        el_status = self.read_mqtt_topic(self.topic_el_status)
        if el_status and 'v' in el_status:
            v = el_status['v']
            logger.info(f"EL: state={v.get('state', 'N/A')}, pos={v.get('act_pos', 'N/A'):.2f}°, target={v.get('target_pos', 'N/A'):.2f}°, in_target={v.get('in_target', 'N/A')}")

    def get_current_mode(self):
        """
        Get current mode from the servo system.
        """
        az_status = self.read_mqtt_topic(self.topic_az_status)
        el_status = self.read_mqtt_topic(self.topic_el_status)

        # Check for 'state' field (which is what the actual MQTT response uses)
        if az_status and 'v' in az_status and 'state' in az_status['v']:
            mode_az = az_status['v']['state']
            # Map numeric mode to string mode using config
            mode_az = config.mqtt.mode_mapping.get(mode_az, str(mode_az))
        else:
            logger.warning(f"{Colors.RED}Warning: Could not read current azimuth state{Colors.RESET}")
            mode_az = None

        if el_status and 'v' in el_status and 'state' in el_status['v']:
            mode_el = el_status['v']['state']
            # Map numeric mode to string mode using config
            mode_el = config.mqtt.mode_mapping.get(mode_el, str(mode_el))
        else:
            logger.warning(f"{Colors.RED}Warning: Could not read current elevation state{Colors.RESET}")
            mode_el = None

        return mode_az, mode_el

    def get_current_position(self):
        """
        Get current telescope position from the servo system.
        
        Returns:
            tuple: (current_az, current_el) in degrees, or (None, None) if unavailable
        """
        try:
            # Read azimuth position
            az_status = self.read_mqtt_topic(self.topic_az_status, timeout=0.5)
            if az_status and 'v' in az_status and 'act_pos' in az_status['v']:
                current_az = az_status['v']['act_pos']
            else:
                logger.warning(f"{Colors.RED}Warning: Could not read current azimuth position{Colors.RESET}")
                return None, None
                
            # Read elevation position  
            el_status = self.read_mqtt_topic(self.topic_el_status, timeout=0.5)
            if el_status and 'v' in el_status and 'act_pos' in el_status['v']:
                current_el = el_status['v']['act_pos']
            else:
                logger.warning(f"{Colors.RED}Warning: Could not read current elevation position{Colors.RESET}")
                return None, None
                
            return float(current_az), float(current_el)
            
        except Exception as e:
            logger.error(f"{Colors.RED}Error reading telescope position: {e}{Colors.RESET}")
            return None, None
        
    def setup_program_track(self):
        """
        Initialize the program track for trajectory following.
        
        Clears any existing program track definition and sets the interpolation
        mode to '1' (linear interpolation between points).
        """
        # Clear program track definition
        sv = {
            'id_client': config.mqtt.id_client_motion,
            'source': config.mqtt.source,
            'destination': config.mqtt.destination_prg_trk,
            'name': 'clear_definition',
            'args': []
        }
        self.send_mqtt_command(self.topic_prg_trk_cmd, sv)
        
        # Set interpolation
        sv = {
            'id_client': config.mqtt.id_client_motion,
            'source': config.mqtt.source,
            'destination': config.mqtt.destination_prg_trk,
            'name': 'set_interpolation',
            'args': ['1']
        }
        self.send_mqtt_command(self.topic_prg_trk_cmd, sv)

    def send_track_start_time(self, time):
        """
        Send the start time to the program track.
        """
        sv = {
            'id_client': config.mqtt.id_client_motion,
            'source': config.mqtt.source,
            'destination': config.mqtt.destination_prg_trk,
            'name': 'set_start_time',
            'args': [time]
        }
        self.send_mqtt_command(self.topic_prg_trk_cmd, sv)

    def send_track_position(self, rt, mnt_az, mnt_el, ds):
        """
        Send a position command to the program track.
        """
        sv = {
            'id_client': config.mqtt.id_client_tracking,
            'source': config.mqtt.source,
            'destination': config.mqtt.destination_prg_trk,
            'name': 'append_entries',
            'args': [[rt], [mnt_az], [mnt_el]]
        }
        
        # Wait until datetime ds to send position
        while (datetime.now(timezone.utc) - ds).total_seconds() < 0:
            time.sleep(config.mqtt.track_sleep_interval)
            
        # Send position to servo
        self.send_mqtt_command(self.topic_prg_trk_cmd, sv)