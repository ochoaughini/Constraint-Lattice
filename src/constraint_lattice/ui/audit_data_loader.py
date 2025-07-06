# SPDX-License-Identifier: MIT
# Copyright (c) 2025 ochoaughini. All rights reserved.
# See LICENSE for full terms.


class AuditDataLoader:
    """Loads audit data from files and Kafka"""
    
    def __init__(self):
        self.sessions = {}
        self.queue = Queue()
        self.consumer = None
        
        # Start Kafka consumer if enabled
        if os.getenv("ENABLE_KAFKA_SINK", "false").lower() == "true":
            self._start_kafka_consumer()
    
    def _start_kafka_consumer(self):
        """Start Kafka consumer in background thread"""
        conf = {
            'bootstrap.servers': os.getenv("KAFKA_SERVERS", "localhost:9092"),
            'group.id': 'audit-panel',
            'auto.offset.reset': 'earliest'
        }
        self.consumer = Consumer(conf)
        self.consumer.subscribe([os.getenv("KAFKA_TOPIC", "constraint_audit")])
        
        # Start consumer thread
        threading.Thread(target=self._consume_messages, daemon=True).start()
    
    def _consume_messages(self):
        """Consume messages from Kafka"""
        while True:
            msg = self.consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                raise KafkaException(msg.error())
            
            # Parse message
            try:
                audit_step = json.loads(msg.value().decode('utf-8'))
                self.queue.put(audit_step)
            except Exception as e:
                print(f"Error processing message: {e}")
    
    def load_from_file(self, path: str):
        """Load audit data from file"""
        with open(path, 'r') as f:
            data = json.load(f)
            
        # Group by session
        for step in data:
            session_id = step.get("session_id")
            if session_id not in self.sessions:
                self.sessions[session_id] = []
            self.sessions[session_id].append(step)
        
        # Sort each session by timestamp
        for session in self.sessions.values():
            session.sort(key=lambda x: x["timestamp_utc"])
    
    def get_sessions(self) -> List[str]:
        """Get list of session IDs"""
        return list(self.sessions.keys())
    
    def get_session_steps(self, session_id: str) -> List[Dict]:
        """Get audit steps for a session"""
        return self.sessions.get(session_id, [])
    
    def get_new_steps(self):
        """Get new steps from the queue"""
        steps = []
        while not self.queue.empty():
            steps.append(self.queue.get())
        return steps
