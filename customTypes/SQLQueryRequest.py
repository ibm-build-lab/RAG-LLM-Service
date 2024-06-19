from pydantic import BaseModel

class SQLQuery(BaseModel):
    query: str