from sqlalchemy import Integer, String, Float, Boolean, Date, ForeignKey, Column, UniqueConstraint, DateTime, JSON
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()

class PlatformDimension(Base):
    __tablename__ = 'platform_dimension'
    platform_id = Column(Integer, primary_key=True, autoincrement=True)
    platform_name = Column(String(255), nullable=False)
    order = relationship('OrderFact', back_populates='platform')
    
class DateDimension(Base):
    __tablename__ = 'date_dimension'
    date_id = Column(Integer, primary_key=True, autoincrement=False)
    date = Column(Date, nullable=False)
    days_name = Column(String(50), nullable=False)
    month = Column(String(50), nullable=False)
    year = Column(Integer, nullable=False)
    is_weekend = Column(Boolean, default=False)
    is_twin_date = Column(Boolean, default=False)
    is_payday = Column(Boolean, default=False)
    is_ramadhan = Column(Boolean, default=False)
    order = relationship('OrderFact', back_populates='date')
    
class PaymentMethodDimension(Base):
    __tablename__ = 'payment_method_dimension'
    payment_method_id = Column(Integer, primary_key=True, autoincrement=True)
    payment_method_name = Column(String(50), nullable=False)
    payment_method_category = Column(String(50),nullable=False)
    order = relationship('OrderFact', back_populates='payment_method')
    
class ProductDimension(Base):
    __tablename__ = 'product_dimension'
    product_id = Column(Integer, primary_key=True, autoincrement=True)
    product_model = Column(String(50), nullable=False)
    product_color = Column(String(50), nullable=False)
    product_size = Column(String(50), nullable=False)
    is_muslim_fashion = Column(Boolean, default=False)
    order = relationship('OrderFact', back_populates='product')
    
class LocationDimension(Base):
    __tablename__ = 'location_dimension'
    location_id = Column(Integer, primary_key=True, autoincrement=True)
    province = Column(String(50), nullable=False)
    city = Column(String(50), nullable=False)
    order = relationship('OrderFact', back_populates='location')
    
class OrderFact(Base):
    __tablename__ = 'order_fact'
    order_id = Column(Integer, primary_key=True, autoincrement=True)
    order_key = Column(String(50), nullable=False)
    date_id = Column(Integer, ForeignKey('date_dimension.date_id'))
    product_id = Column(Integer, ForeignKey('product_dimension.product_id'))
    platform_id = Column(Integer, ForeignKey('platform_dimension.platform_id'))
    payment_method_id = Column(Integer, ForeignKey('payment_method_dimension.payment_method_id'))
    location_id = Column(Integer, ForeignKey('location_dimension.location_id'))
    status = Column(String(20), nullable=False)
    cancel_reason = Column(String(255), nullable=True)
    quantity = Column(Integer, nullable=False)
    price = Column(Float, nullable=False)
    discount = Column(Float, default=0.0)
    total_amount = Column(Float, nullable=False)
    
    date = relationship('DateDimension', back_populates='order')
    product = relationship('ProductDimension', back_populates='order')
    platform = relationship('PlatformDimension', back_populates='order')
    payment_method = relationship('PaymentMethodDimension', back_populates='order')
    location = relationship('LocationDimension', back_populates='order')
    
    __table_args__ = (
        UniqueConstraint('order_key', 'product_id', name='uq_order_product'),
    )
    
class User(Base):
    __tablename__ = "users"
    __table_args__ = {'schema': 'lamiah_app'}
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, nullable=False)
    role = Column(String(50), default="admin")
    created_at = Column(DateTime, default=func.now())
    
class FactDailyAgregat(Base):
    __tablename__ = 'fact_daily_agregat'
    
    agregat_id = Column(Integer, primary_key=True, autoincrement=True)
    date_id = Column(Integer, ForeignKey('date_dimension.date_id'))
    product_id = Column(Integer, ForeignKey('product_dimension.product_id'))
    status = Column(String(20), nullable=False)
    
    total_orders = Column(Integer, nullable=False, default=0)
    total_quantity = Column(Integer, nullable=False, default=0)
    total_amount = Column(Float, nullable=False, default=0.0)
    
    __table_args__ = (
        UniqueConstraint('date_id', 'product_id', 'status', name='uq_daily_agregat'),
    )

class ForecastCache(Base):
    __tablename__ = 'forecast_cache'
    __table_args__ = {'schema': 'lamiah_app'}
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=func.now())
    db_state = Column(String(100), nullable=False)
    total_forecast_qty = Column(Integer, nullable=False)
    forecast_daily = Column(JSON, nullable=False)
    top_models = Column(JSON, nullable=False)
    top_colors = Column(JSON, nullable=False)