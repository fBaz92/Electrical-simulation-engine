class PIController:
    """
    Proportional-Integral (PI) controller with output saturation.
    
    Implements a discrete-time PI controller with the control law:
    u(t) = Kp * e(t) + Ki * ∫e(τ)dτ
    
    where e(t) is the error signal and the integral is computed numerically
    using forward Euler integration.
    
    The controller output is clamped between u_min and u_max to prevent
    windup and ensure realistic control signals.
    
    Attributes:
        Kp: Proportional gain.
        Ki: Integral gain.
        u_min: Minimum output value (lower saturation limit).
        u_max: Maximum output value (upper saturation limit).
        integrator: Internal integrator state (accumulated integral term).
    """
    def __init__(self, Kp, Ki, u_min, u_max, antiwindup=True):
        """
        Initialize the PI controller.
        
        Args:
            Kp: Proportional gain. Determines the response to current error.
            Ki: Integral gain. Determines the response to accumulated error over time.
            u_min: Minimum output value. Control signal will be clamped to this value
                  if it falls below it.
            u_max: Maximum output value. Control signal will be clamped to this value
                  if it exceeds it.
            antiwindup: If True, the integrator will be back calculated when the output is saturated.
                This is useful to prevent the integrator from growing indefinitely when the output is saturated.
        """
        self.Kp = Kp
        self.Ki = Ki
        self.u_min = u_min
        self.u_max = u_max
        self.antiwindup = antiwindup
        self.integrator = 0.0

    def update(self, error, dt):
        """
        Update the controller and compute the control output.
        
        Computes the PI control signal based on the current error and updates
        the internal integrator state. The output is saturated between u_min
        and u_max.
        
        The control law is:
        u = Kp * error + integrator
        
        where integrator is updated as:
        integrator += Ki * error * dt
        
        Args:
            error: Current error signal (reference - measured value).
            dt: Time step for numerical integration in seconds.
        
        Returns:
            Control output signal u, clamped between u_min and u_max.
        """
        self.integrator += self.Ki * error * dt
        u = self.Kp * error + self.integrator

        # saturazione base
        if u > self.u_max:
            u = self.u_max
        elif u < self.u_min:
            u = self.u_min

        if self.antiwindup:
            # back calculation of the integrator
            self.integrator = u - self.Kp * error
            if self.integrator > self.u_max:
                self.integrator = self.u_max
            elif self.integrator < self.u_min:
                self.integrator = self.u_min

        return u
