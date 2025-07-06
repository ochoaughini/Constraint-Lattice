# SPDX-License-Identifier: MIT
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
import logging

# Create a shared logger instance
logger = logging.getLogger('varkiel')
logger.setLevel(logging.INFO)
handler = logging.FileHandler('varkiel.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class ReflectiveLogger:
    def __init__(self):
        self.logger = logging.getLogger('reflective')
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler('reflective.log')
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
    def log_decision(self, decision_data):
        self.logger.info(decision_data)
