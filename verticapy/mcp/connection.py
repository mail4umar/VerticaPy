import verticapy as vp

# -----------------------------
# Connection setup (run once)
# -----------------------------

# CONN_INFO = {
#     "host": "10.10.10.235",
#     "port": "34101",
#     "database": "verticadb21477",
#     "password": "",
#     "user": "ughumman",
# }

CONN_INFO = {
    "host": "10.10.10.69",
    "port": "32796",
    "database": "verticadb21477",
    "password": "",
    "user": "ughumman",
}

CONN_INFO = {
   "host": "127.0.0.1",      # or "host.docker.internal" if using Docker Desktop
   "port": 5433,
   "database": "demo",       # or your created DB like "verticadb21477"
   "user": "dbadmin",
   "password": "",            # if no password set
   "tls_verify": False
}

class VerticaPyConnection:
    """Manages VerticaPy database connections."""
    
    def __init__(self):
        self.is_connected = False
        self.connection_name = "VerticaDSN"
    
    def connect(self) -> tuple[bool, str]:
        """Establish connection to Vertica database."""
        try:
            vp.new_connection(
                CONN_INFO,
                name=self.connection_name,
                auto=True,
                overwrite=True,
            )
            self.is_connected = True
            return True, "Successfully connected to Vertica database"
        except Exception as e:
            error_msg = f"Connection failed: {str(e)}"
            print(error_msg)
            self.is_connected = False
            return False, error_msg
    
    def ensure_connected(self) -> tuple[bool, str]:
        """Ensure we have an active connection."""
        if not self.is_connected:
            return self.connect()
        return True, "Already connected"
    
    def disconnect(self) -> tuple[bool, str]:
        """Disconnect from Vertica database."""
        try:
            if self.is_connected:
                vp.close_connection(self.connection_name)
                self.is_connected = False
                return True, "Successfully disconnected from Vertica database"
            else:
                return True, "Already disconnected"
        except Exception as e:
            error_msg = f"Disconnect failed: {str(e)}"
            print(error_msg)
            return False, error_msg
    
    def get_connection_status(self) -> dict:
        """Get current connection status and info."""
        return {
            "is_connected": self.is_connected,
            "connection_name": self.connection_name,
            "host": CONN_INFO["host"],
            "port": CONN_INFO["port"],
            "database": CONN_INFO["database"],
            "user": CONN_INFO["user"]
        }