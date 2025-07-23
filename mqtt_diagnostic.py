#!/usr/bin/env python3
"""
MQTT Diagnostic Script
Test MQTT broker connectivity and responsiveness
"""

import paho.mqtt.client as mqtt
import json
import time
import logging
import threading
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MQTTDiagnostic:
    def __init__(self, broker_ip, port=1883, client_id="diagnostic_client"):
        self.broker_ip = broker_ip
        self.port = port
        self.client_id = client_id
        self.connected = False
        self.messages_received = 0
        self.last_message_time = None
        self.connection_errors = 0
        self.publish_errors = 0
        
        self.client = mqtt.Client(client_id=client_id)
        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect
        self.client.on_message = self.on_message
        self.client.on_publish = self.on_publish
        
    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            logger.info(f"✓ Connected to {self.broker_ip}:{self.port} successfully")
            self.connected = True
            # Subscribe to status topics
            client.subscribe("mtexControls/Motion/MotionAxis/Azimuth/status")
            client.subscribe("mtexControls/Motion/MotionAxis/Elevation/status")
            logger.info("✓ Subscribed to status topics")
        else:
            logger.error(f"✗ Connection failed with code {rc}")
            self.connection_errors += 1
            self.connected = False
            
    def on_disconnect(self, client, userdata, rc):
        logger.warning(f"⚠ Disconnected from broker with code {rc}")
        self.connected = False
        
    def on_message(self, client, userdata, msg):
        self.messages_received += 1
        self.last_message_time = datetime.now()
        try:
            data = json.loads(msg.payload.decode())
            logger.info(f"📨 Received message on {msg.topic}: {len(msg.payload)} bytes")
        except json.JSONDecodeError:
            logger.warning(f"⚠ Invalid JSON on {msg.topic}")
            
    def on_publish(self, client, userdata, mid):
        logger.info(f"✓ Message {mid} published successfully")
        
    def connect(self):
        try:
            logger.info(f"Connecting to {self.broker_ip}:{self.port}...")
            self.client.connect(self.broker_ip, self.port, 60)
            self.client.loop_start()
            
            # Wait for connection
            timeout = 10
            start_time = time.time()
            while not self.connected and (time.time() - start_time) < timeout:
                time.sleep(0.1)
                
            if not self.connected:
                logger.error(f"✗ Failed to connect within {timeout}s")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"✗ Connection error: {e}")
            self.connection_errors += 1
            return False
            
    def test_publish(self):
        """Test publishing a simple message"""
        if not self.connected:
            logger.error("✗ Not connected, cannot test publish")
            return False
            
        try:
            test_msg = {"test": "diagnostic", "timestamp": datetime.now().isoformat()}
            result = self.client.publish("test/diagnostic", json.dumps(test_msg))
            
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                logger.info("✓ Test publish successful")
                return True
            else:
                logger.error(f"✗ Publish failed with code {result.rc}")
                self.publish_errors += 1
                return False
                
        except Exception as e:
            logger.error(f"✗ Publish error: {e}")
            self.publish_errors += 1
            return False
            
    def run_diagnostic(self, duration=30):
        """Run diagnostic for specified duration"""
        logger.info(f"🔧 Starting {duration}s diagnostic test...")
        
        if not self.connect():
            return False
            
        start_time = time.time()
        last_stats_time = start_time
        
        while (time.time() - start_time) < duration:
            # Test publish every 5 seconds
            if (time.time() - last_stats_time) >= 5:
                self.test_publish()
                logger.info(f"📊 Stats: {self.messages_received} messages received, "
                          f"{self.connection_errors} connection errors, "
                          f"{self.publish_errors} publish errors")
                last_stats_time = time.time()
                
            time.sleep(1)
            
        logger.info("🏁 Diagnostic complete")
        self.client.loop_stop()
        self.client.disconnect()
        
        return True
        
    def print_summary(self):
        logger.info("=" * 50)
        logger.info("DIAGNOSTIC SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Messages received: {self.messages_received}")
        logger.info(f"Connection errors: {self.connection_errors}")
        logger.info(f"Publish errors: {self.publish_errors}")
        if self.last_message_time:
            logger.info(f"Last message: {self.last_message_time}")
        else:
            logger.warning("No messages received!")

def main():
    # Test both brokers
    brokers = [
        ("North Antenna", "192.168.65.60"),
        ("South Antenna", "192.168.65.50")
    ]
    
    for name, ip in brokers:
        logger.info(f"\n{'='*20} Testing {name} ({ip}) {'='*20}")
        
        # Use unique client ID to avoid conflicts
        client_id = f"diagnostic_{name.lower().replace(' ', '_')}_{int(time.time())}"
        diagnostic = MQTTDiagnostic(ip, client_id=client_id)
        
        diagnostic.run_diagnostic(15)  # 15 second test
        diagnostic.print_summary()
        
        time.sleep(2)  # Brief pause between tests

if __name__ == "__main__":
    main() 